"""Generic looped-transformer runner (harness v1.1).

A network is a folder holding network.json (config), weights.npz and tokenizer.py (the network's own generic
tokenizer). The runner does only this, for any network:

  T = tokenize(text)                              (n, F) integer fields per token; text is passed in unchanged
  h = sum over fields f of E_f[T[:, f]]           embedding: one table per field
  repeat R times (R from the config's rule; it depends only on n), for each layer of the one shared block:
      h += concat_heads(softmax(q k^T / sqrt(d_k) + mask) v) W_o    mask per head: "none" or "causal"; no norm
      h += relu(h W1 + b1) W2 + b2
  logits = h W_out + b_out                        at every position
  response = "".join(alphabet[argmax logits[i]] for i in positions)   readout "emit"

Readout "emit": the alphabet is a list of strings declared in network.json. At each position the network writes the
string with the highest logit (ties: the lowest index); the empty string writes nothing. The response is the written
strings in position order with nothing added between, before or after them, and it is returned unchanged.

  python -I runner/looped.py <network_dir> <text_file> [--backend numpy|numpy32|torch32|torch64]
"""
import argparse
import importlib.util
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import flops as FL  # noqa: E402

BLOCK = 2048          # most query rows per attention chunk in the exact (non-fused) paths
CHUNK = 2 ** 25       # most score entries (rows x keys) per chunk, so long prompts fit in memory


BACKENDS = ["numpy", "numpy32", "torch32", "torch64"]


def _rows(n_keys):
    return max(1, min(BLOCK, CHUNK // n_keys))


def softmax(s):
    s = s - s.max(axis=-1, keepdims=True)
    e = np.exp(s)
    return e / e.sum(axis=-1, keepdims=True)


def load(net_dir):
    cfg = json.load(open(os.path.join(net_dir, "network.json")))
    z = np.load(os.path.join(net_dir, "weights.npz"))

    def arr(k):
        if k in z:
            return np.asarray(z[k], dtype=np.float64)
        if k + ".shape" in z:                      # sparse storage: flat indices and values
            shape = tuple(int(s) for s in z[k + ".shape"])
            a = np.zeros(math.prod(shape))
            a[z[k + ".idx"].astype(np.int64)] = z[k + ".val"]
            return a.reshape(shape)
        raise KeyError(k)

    d = cfg["d_model"]
    net = {"cfg": cfg, "emb": [arr(f"emb.{f}") for f in range(len(cfg["fields"]))],
           "W_out": arr("out.W"), "b_out": arr("out.b"), "layers": []}
    for l, L in enumerate(cfg["layers"]):
        heads = L["heads"]
        sk, sv = sum(h["dk"] for h in heads), sum(h["dv"] for h in heads)
        lay = {"heads": heads, "W1": arr(f"L{l}.W1"), "b1": arr(f"L{l}.b1"), "W2": arr(f"L{l}.W2"),
               "b2": arr(f"L{l}.b2")}
        if heads:
            lay.update({k: arr(f"L{l}.{k}") for k in ("Wq", "Wk", "Wv", "Wo")})
            assert lay["Wq"].shape == (d, sk) and lay["Wk"].shape == (d, sk)
            assert lay["Wv"].shape == (d, sv) and lay["Wo"].shape == (sv, d)
        assert lay["W1"].shape == (d, L["d_ff"]) and lay["W2"].shape == (L["d_ff"], d)
        net["layers"].append(lay)
    assert all(E.shape[1] == d for E in net["emb"])
    assert net["W_out"].shape == (d, len(cfg["alphabet"])) and net["b_out"].shape == (len(cfg["alphabet"]),)
    return net


def load_tokenizer(net_dir):
    spec = importlib.util.spec_from_file_location("net_tokenizer", os.path.join(net_dir, "tokenizer.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.tokenize


def loops(rule, n):
    return rule["const"] + rule["lin"] * -(-n // rule["div"]) + rule["log2"] * math.ceil(math.log2(max(n, 2)))


def tokens(net, tokenize, text):
    T = np.asarray(tokenize(text), dtype=np.int64)
    assert T.ndim == 2 and T.shape[1] == len(net["emb"]), "tokenize must return an (n, F) array of ints"
    for f, E in enumerate(net["emb"]):
        assert T[:, f].min() >= 0 and T[:, f].max() < E.shape[0], f"field {f} out of its table's range"
    return T


def _attend_np(q, k, v, causal, dk):
    """Exact softmax attention in chunks of query rows, in q's dtype. float64 scales q by 1/sqrt(dk) first, as
    torch64 does; float32 scales the scores, as the fused kernel's scale argument does."""
    out = np.empty((q.shape[0], v.shape[1]), dtype=q.dtype)
    step = _rows(k.shape[0])
    for a in range(0, q.shape[0], step):
        b = min(a + step, q.shape[0])
        if q.dtype == np.float32:
            s = (q[a:b] @ k.T) * np.float32(1.0 / math.sqrt(dk))
        else:
            s = (q[a:b] / np.sqrt(dk)) @ k.T
        if causal:
            s[np.arange(b - a)[:, None] + a < np.arange(k.shape[0])[None, :]] = -np.inf
        out[a:b] = softmax(s) @ v
    return out


def forward_numpy(net, T, R, dt=np.float64):
    """numpy: float64. numpy32: every weight, activation and score in float32, following the GPU path (torch32) as
    closely as numpy can; the fused kernel's own order of summation can still differ in the last bits."""
    c = lambda a: np.asarray(a, dtype=dt)
    E = [c(e) for e in net["emb"]]
    Ls = [{k: (c(v) if isinstance(v, np.ndarray) else v) for k, v in L.items()} for L in net["layers"]]
    h = E[0][T[:, 0]].copy()
    for f in range(1, T.shape[1]):
        h += E[f][T[:, f]]
    for _ in range(R):
        for L in Ls:
            if L["heads"]:
                Q, K, V = h @ L["Wq"], h @ L["Wk"], h @ L["Wv"]
                outs, q0, v0 = [], 0, 0
                for hd in L["heads"]:
                    dk, dv = hd["dk"], hd["dv"]
                    outs.append(_attend_np(Q[:, q0:q0 + dk], K[:, q0:q0 + dk], V[:, v0:v0 + dv],
                                           hd["mask"] == "causal", dk))
                    q0, v0 = q0 + dk, v0 + dv
                h = h + np.hstack(outs) @ L["Wo"]
            h = h + np.maximum(h @ L["W1"] + L["b1"], 0) @ L["W2"] + L["b2"]
    return (h @ c(net["W_out"]) + c(net["b_out"])).astype(np.float64)


def forward_torch(net, T, R, dtype_name):
    """torch32: float32 with the fused scaled_dot_product_attention kernel, pinned to the memory-efficient kernel so
    the result does not depend on which kernel torch picks (TF32 off). torch64: float64, exact chunked softmax as in
    numpy. Head dims are zero-padded to a multiple of 8 for the fused kernel; this changes no score or output because
    the padding is zero."""
    import torch
    import torch.nn.functional as F
    from torch.nn.attention import SDPBackend, sdpa_kernel
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dt = torch.float32 if dtype_name == "torch32" else torch.float64
    t = lambda a: torch.as_tensor(a, dtype=dt, device=dev)
    Tt = torch.as_tensor(T, device=dev)
    h = t(net["emb"][0])[Tt[:, 0]]
    for f in range(1, T.shape[1]):
        h = h + t(net["emb"][f])[Tt[:, f]]
    Ls = [{k: (t(v) if isinstance(v, np.ndarray) else v) for k, v in L.items()} for L in net["layers"]]
    pad8 = lambda x: F.pad(x, (0, (-x.shape[-1]) % 8))
    with torch.no_grad():
        for _ in range(R):
            for L in Ls:
                if L["heads"]:
                    Q, K, V = h @ L["Wq"], h @ L["Wk"], h @ L["Wv"]
                    outs, q0, v0 = [], 0, 0
                    for hd in L["heads"]:
                        dk, dv, causal = hd["dk"], hd["dv"], hd["mask"] == "causal"
                        q, k, v = Q[:, q0:q0 + dk], K[:, q0:q0 + dk], V[:, v0:v0 + dv]
                        if dtype_name == "torch32":
                            with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
                                o = F.scaled_dot_product_attention(pad8(q)[None, None], pad8(k)[None, None],
                                                                   pad8(v)[None, None], is_causal=causal,
                                                                   scale=1.0 / math.sqrt(dk))[0, 0, :, :dv]
                        else:
                            o = torch.empty((q.shape[0], dv), dtype=dt, device=dev)
                            step = _rows(k.shape[0])
                            for a in range(0, q.shape[0], step):
                                b = min(a + step, q.shape[0])
                                s = (q[a:b] / math.sqrt(dk)) @ k.T
                                if causal:
                                    s = s.masked_fill(torch.arange(a, b, device=dev)[:, None]
                                                      < torch.arange(k.shape[0], device=dev)[None, :], -math.inf)
                                s = s - s.max(dim=-1, keepdim=True).values
                                e = torch.exp(s)
                                o[a:b] = (e / e.sum(dim=-1, keepdim=True)) @ v
                        outs.append(o)
                        q0, v0 = q0 + dk, v0 + dv
                    h = h + torch.cat(outs, dim=1) @ L["Wo"]
                h = h + torch.relu(h @ L["W1"] + L["b1"]) @ L["W2"] + L["b2"]
        return (h @ t(net["W_out"]) + t(net["b_out"])).double().cpu().numpy()


def emit(net, logits):
    alphabet = net["cfg"]["alphabet"]
    return "".join(alphabet[i] for i in logits.argmax(axis=-1))


def run(net, tokenize, text, backend="numpy"):
    """text -> (response, info). Refuses a prompt whose counted FLOPs exceed the cap."""
    T = tokens(net, tokenize, text)
    n = len(T)
    R = loops(net["cfg"]["loops"], n)
    fl = FL.count(net["cfg"], n, R)
    info = {"n_chars": len(text), "n_tokens": n, "R": R, "flops": fl, "backend": backend}
    if fl > FL.CAP:
        raise RuntimeError(f"refused: {fl:.3e} FLOPs > cap {FL.CAP:.0e} (n = {n}, R = {R})")
    if backend in ("numpy", "numpy32"):
        logits = forward_numpy(net, T, R, np.float32 if backend == "numpy32" else np.float64)
    else:
        logits = forward_torch(net, T, R, backend)
    return emit(net, logits), info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("net_dir")
    ap.add_argument("text_file")
    ap.add_argument("--backend", default="numpy", choices=BACKENDS)
    a = ap.parse_args()
    net = load(a.net_dir)
    text = open(a.text_file, encoding="utf-8", newline="").read()
    resp, info = run(net, load_tokenizer(a.net_dir), text, a.backend)
    print(json.dumps(info), file=sys.stderr)
    sys.stdout.write(resp)


if __name__ == "__main__":
    main()
