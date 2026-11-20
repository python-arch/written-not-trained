"""Write every weight of the Maze-Hard looped transformer (nothing is trained).

    python3 -I work/write_weights.py          -> work/net/{network.json, weights.npz} (tokenizer.py is hand-written)

What the network computes (one token per byte of the 30 x 30 maze text, rows 31 tokens apart):
  1. a flood from G: a cell becomes "reached" (r = 1) one loop after an open neighbour is reached, so cells are
     reached in order of their BFS distance to G;
  2. at the loop a cell is reached it fixes its pointer P = the first direction in the order Down, Left, Right, Up
     whose neighbour was reached one loop earlier (one step closer to G);
  3. a chain of marks from S: S starts marked; a marked cell exposes "mark-out" in its pointer direction, and a
     cell whose neighbour's mark-out points at it is marked one loop later.
  The marked cells are the lexicographically smallest (D < L < R < U) shortest S -> G path, the DEV convention.
Neighbours are read by attention on residue-phase position codes (positions mod 9, 10, 11) rotated by the offset
(+31, -1, +1, -31). Where the neighbour does not exist (row 0 up, row 29 down, the last token right), a flag turns on
a large score bonus on the newline tokens, which act as walls. The left head is causal, so token 0 reads itself.
"""
import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "net")

MODULI = (9, 10, 11)
WMOD = np.array([0.27, 0.332, 0.398])          # weights of the moduli in the position kernel (sum 1)
W_ROW = 31                                      # tokens per maze line (30 cells + newline)
N_TOK = 30 * W_ROW - 1                          # 929 tokens in a maze prompt
LAST = N_TOK - 1

# ---- residual dimensions ----
CONST, WALL, NLD, SG = 0, 1, 2, 3
PH = 4                                          # 4..9: cos/sin for each modulus
R = 10                                          # reached flag
RD, RL, RR, RU = 11, 12, 13, 14                 # accumulated reads of the neighbours' r
PD, PL, PR, PU = 15, 16, 17, 18                 # pointer (one-hot)
OD, OL, OR, OU = 19, 20, 21, 22                 # mark-out in each direction
IN = 23                                         # accumulated incoming marks (>= 1: on the path)
INVUP, AACC, INVDN, ISLAST = 24, 25, 26, 27     # boundary flags (accumulated over loops)
D = 28

ALPHABET = ["", "#", " ", "o", "S", "G", "\n"]


def phase(i):
    return np.concatenate([[math.cos(2 * math.pi * i / m), math.sin(2 * math.pi * i / m)] for m in MODULI])


def kernel(x):
    return sum(w * np.cos(2 * np.pi * np.asarray(x) / m) for w, m in zip(WMOD, MODULI))


# margin of the position kernel: 1 at offset 0, at most 1 - KM at every other offset that can occur
KM = 1.0 - kernel(np.arange(1, N_TOK + W_ROW + 1)).max()
assert KM > 0.06, KM
BETA = 900.0                                    # BETA * KM ~ 57: off-target attention weight < 1e-22
GAMMA = 2 * BETA + 100.0                        # sink bonus (per unit of flag) beats any kernel score
BIG = 1.0e4                                     # gate constant; every counter stays far below it

# value read by the causal newline head; its mean over the newlines seen so far is < 0 above the last row, > 0 in
# it (a fixed property of the 30 x 31 layout, checked below). Found once by a small LP on the layout geometry.
WA = np.array([-1.0, -1.0, -0.7585, -1.0, 0.6033, -0.4154])
WA0 = 0.0031


def a_value(j):
    return float(WA @ phase(j) + WA0)


def check_layout():
    nls = [W_ROW * k + W_ROW - 1 for k in range(29)]
    above = [np.mean([a_value(j) for j in range(i + 1)]) for i in range(30)]          # row 0: no newline before
    above += [np.mean([a_value(j) for j in nls[:r]]) for r in range(1, 29)]           # rows 1..28
    last = np.mean([a_value(j) for j in nls])                                         # row 29
    assert max(above) < -0.02 and last > 0.02, (max(above), last)
    return min(-max(above), last), last


def build():
    amarg, a29 = check_layout()
    ascale = 1.0 / amarg                          # A <= -1 above the last row, >= 1 in it
    W = {}

    # ---- embedding: 8 bit fields, then position mod 9, 10, 11 ----
    for k in range(8):
        W[f"emb.{k}"] = np.zeros((2, D))
    W["emb.0"][:, CONST] = 1.0
    W["emb.0"][1, WALL] += 1.0                    # '#'=0x23 ' '=0x20 'S'=0x53 'G'=0x47 '\n'=0x0A
    W["emb.4"][1, WALL] -= 1.0
    W["emb.2"][1, WALL] -= 1.0
    W["emb.3"][1, NLD] = 1.0
    W["emb.4"][1, SG] = 1.0
    W["emb.2"][1, SG] = -1.0
    W["emb.2"][1, R] = 1.0                        # G starts reached
    W["emb.4"][1, IN] = 1.0                       # S starts marked
    for t, m in enumerate(MODULI):
        E = np.zeros((m, D))
        for r in range(m):
            E[r, PH + 2 * t] = math.cos(2 * math.pi * r / m)
            E[r, PH + 2 * t + 1] = math.sin(2 * math.pi * r / m)
        W[f"emb.{8 + t}"] = E

    # ---- layer 0: causal newline head (flags) + 2-unit MLP ----
    Wq = np.zeros((D, 1)); Wk = np.zeros((D, 1)); Wv = np.zeros((D, 2)); Wo = np.zeros((2, D))
    Wq[CONST, 0] = 60.0
    Wk[NLD, 0] = 1.0
    Wv[CONST, 0] = 1.0; Wv[NLD, 0] = -1.0         # 1 if no newline before (row 0), else 0
    Wv[PH:PH + 6, 1] = WA * ascale; Wv[CONST, 1] = WA0 * ascale
    Wo[0, INVUP] = 1.0; Wo[1, AACC] = 1.0
    W.update({"L0.Wq": Wq, "L0.Wk": Wk, "L0.Wv": Wv, "L0.Wo": Wo})
    W1 = np.zeros((D, 2)); b1 = np.zeros(2); W2 = np.zeros((2, D))
    W1[AACC, 0] = 1.0 / (a29 * ascale)            # row 29: += loop index; other rows: 0
    W2[0, INVDN] = 1.0
    c = 2.0 / KM                                  # 1 at the last token, 0 elsewhere
    W1[PH:PH + 6, 1] = c * np.concatenate([w * phase(LAST)[2 * t:2 * t + 2] for t, w in enumerate(WMOD)])
    b1[1] = -c * (1.0 - KM / 2)
    W2[1, ISLAST] = 1.0
    W.update({"L0.W1": W1, "L0.b1": b1, "L0.W2": W2, "L0.b2": np.zeros(D)})

    # ---- layer 1: four neighbour heads + 10-unit MLP ----
    # (offset, causal, sink flag dim or None, accumulator of r, mark-out dim read from the neighbour)
    spec = [(+W_ROW, False, INVDN, RD, OU),   # down neighbour; it marks me if it points up
            (-1, True, None, RL, OR),         # left neighbour; points right
            (+1, False, ISLAST, RR, OL),      # right neighbour; points left
            (-W_ROW, False, INVUP, RU, OD)]   # up neighbour; points down
    cfg_heads, Qs, Ks, Vs, Os = [], [], [], [], []
    for off, causal, sink, racc, mdim in spec:
        dk = 6 + (sink is not None)
        q = np.zeros((D, dk)); k = np.zeros((D, dk)); v = np.zeros((D, 2)); o = np.zeros((2, D))
        sq = math.sqrt(dk)
        for t, m in enumerate(MODULI):
            th = 2 * math.pi * off / m
            g = BETA * WMOD[t] * sq
            cd, sd = PH + 2 * t, PH + 2 * t + 1
            q[cd, 2 * t] = math.cos(th) * g; q[sd, 2 * t] = -math.sin(th) * g
            q[sd, 2 * t + 1] = math.cos(th) * g; q[cd, 2 * t + 1] = math.sin(th) * g
            k[cd, 2 * t] = 1.0; k[sd, 2 * t + 1] = 1.0
        if sink is not None:
            q[sink, 6] = GAMMA * sq
            k[NLD, 6] = 1.0
        v[R, 0] = 1.0; o[0, racc] = 1.0
        v[mdim, 1] = 1.0; o[1, IN] = 1.0
        cfg_heads.append({"dk": dk, "dv": 2, "mask": "causal" if causal else "none"})
        Qs.append(q); Ks.append(k); Vs.append(v); Os.append(o)
    W["L1.Wq"] = np.hstack(Qs); W["L1.Wk"] = np.hstack(Ks); W["L1.Wv"] = np.hstack(Vs); W["L1.Wo"] = np.vstack(Os)

    W1 = np.zeros((D, 10)); b1 = np.zeros(10); W2 = np.zeros((10, D))
    racc = [RD, RL, RR, RU]; ptr = [PD, PL, PR, PU]; mout = [OD, OL, OR, OU]
    # reach: r += min(1, 2 * (sum R - 0.5)^+) for an open, not yet reached cell
    for u, thr in ((0, 0.5), (1, 1.0)):
        W1[racc, u] = 1.0
        W1[R, u] = W1[WALL, u] = W1[NLD, u] = -BIG
        b1[u] = -thr
    W2[0, R] = 2.0; W2[1, R] = -2.0
    # pointer, fixed at the loop the cell is reached: first direction whose neighbour was reached one loop earlier
    for dd in range(4):
        u = 2 + dd
        W1[racc[dd], u] = 1.0
        for e in range(dd):
            W1[racc[e], u] = -1.0
        W1[R, u] = W1[WALL, u] = W1[NLD, u] = -BIG
        b1[u] = -0.5
        W2[u, ptr[dd]] = 2.0
    # mark-out: set once, when the cell is marked (IN >= 1) and points this way
    for dd in range(4):
        u = 6 + dd
        W1[IN, u] = 1.0
        W1[ptr[dd], u] = BIG
        W1[mout[dd], u] = -BIG
        b1[u] = -0.5 - BIG
        W2[u, mout[dd]] = 2.0
    W.update({"L1.W1": W1, "L1.b1": b1, "L1.W2": W2, "L1.b2": np.zeros(D)})

    # ---- readout ----
    Wout = np.zeros((D, len(ALPHABET))); bout = np.zeros(len(ALPHABET))
    a = {s: i for i, s in enumerate(ALPHABET)}
    bout[a[""]] = -1.0
    Wout[WALL, a["#"]] = 1.0e5
    bout[a[" "]] = 1.0
    Wout[IN, a["o"]] = 4.0
    Wout[SG, a["S"]] = 1.0e5
    Wout[SG, a["G"]] = -1.0e5
    Wout[NLD, a["\n"]] = 1.0e5           # token 30 has no up target and no sink: its IN is noise
    W["out.W"] = Wout; W["out.b"] = bout

    cfg = {"d_model": D,
           "fields": [f"bit{k}" for k in range(8)] + [f"pos_mod{m}" for m in MODULI],
           "layers": [{"heads": [{"dk": 1, "dv": 2, "mask": "causal"}], "d_ff": 2},
                      {"heads": cfg_heads, "d_ff": 10}],
           "loops": {"const": 0, "lin": 1, "div": 2, "log2": 0},
           "alphabet": ALPHABET}
    return cfg, W


if __name__ == "__main__":
    cfg, W = build()
    os.makedirs(OUT, exist_ok=True)
    json.dump(cfg, open(os.path.join(OUT, "network.json"), "w"), indent=1)
    np.savez(os.path.join(OUT, "weights.npz"), **W)
    total = sum(v.size for v in W.values())
    print(f"wrote {OUT}: {total} weights, {sum(int(np.count_nonzero(v)) for v in W.values())} nonzero;"
          f" kernel margin {KM:.4f}")
