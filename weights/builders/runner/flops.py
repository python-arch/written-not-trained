"""Compute counter for the generic runner (harness v1). Counts dense floating-point operations per prompt from the
network's shapes, n (tokens) and R (loops); a multiply-add counts as 2. Sparsity is not credited, and a causal head is
counted like a full one.

  per loop, per layer:
    projections   2 n d (2 Σd_k + Σd_v)          q, k, v
    scores        2 n² d_k            per head
    softmax       5 n²                per head
    mix           2 n² d_v            per head
    output        2 n (Σd_v) d
    MLP           4 n d d_ff + 3 n d_ff + 2 n d   (two matmuls, bias, relu, bias)
    residuals     2 n d
  once: embedding n d (F - 1); readout 2 n d |alphabet| + n |alphabet|
"""

import json  # noqa: E402
import os  # noqa: E402

_BUILD = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "build.json")
CAP = float(json.load(open(_BUILD))["cap"]) if os.path.exists(_BUILD) else 5e15   # FLOPs per prompt (build.json)


def count(cfg, n, R):
    d = cfg["d_model"]
    per_loop = 0
    for L in cfg["layers"]:
        hs = L["heads"]
        sk, sv = sum(h["dk"] for h in hs), sum(h["dv"] for h in hs)
        if hs:
            per_loop += 2 * n * d * (2 * sk + sv) + 2 * n * sv * d
            per_loop += sum(2 * n * n * h["dk"] + 5 * n * n + 2 * n * n * h["dv"] for h in hs)
            per_loop += 2 * n * d
        per_loop += 4 * n * d * L["d_ff"] + 3 * n * L["d_ff"] + 2 * n * d + 2 * n * d
    a = len(cfg["alphabet"])
    return int(R * per_loop + n * d * (len(cfg["fields"]) - 1) + 2 * n * d * a + n * a)
