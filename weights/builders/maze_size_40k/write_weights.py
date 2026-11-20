"""Writes every weight of the maze network (no training). Run: python3 -I work/write_weights.py

What the network computes, per token (one token per character; the 30x30 grid is 30 lines of 31 characters with
the newline, so the up/down neighbours of position p are p-31 / p+31 and left/right are p-1 / p+1):

  * Position: the tokenizer gives the base-32 digits (lo, hi) of the position. The embedding writes lo, hi,
    lo^2+hi^2, (lo-1)%32, (lo+1)%32, [lo=0], [lo=31] and three edge flags. Four attention heads (up, down, left,
    right) score keys by -beta * ((lo_k - lo_t)^2 + (hi_k - hi_t)^2), which is maximal exactly at the target
    position p+delta (written digit-wise with its carry). When the target is off the grid, the edge flag adds a
    large bonus to newline keys, so the head reads a newline (a never-reached, off-path token).
  * BFS from G: r = reached flag (G starts at 1); every loop an open cell with a reached neighbour becomes reached.
    T counts the loops a token was not reached, so a reached cell holds its distance to G and an unreached token
    holds the loop count (>= the distance of any reached cell).
  * Pointer: once a cell is reached, C_dir = [T - T_neighbour >= 1] marks the neighbours one step closer to G.
  * Path: P starts at 1 on S. A cell on the path exports E_dir = P and the first C in the order Down, Left, Right,
    Up (the dataset's convention). A cell whose neighbour exports E pointing at it joins the path.
  * Readout: walls, newlines, S, G as given; open cells with P = 1 write 'o', else ' '.
All updates are monotone and idempotent, so extra loops change nothing.
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "net")
R_LOOPS = 500

# residual layout
names = ["one", "lo", "hi", "sq", "Lm", "Lp", "c0", "c31", "invU", "invD", "invR", "invL",
         "wall", "nl", "S", "G", "space",
         "r", "T", "P", "C_D", "C_L", "C_R", "C_U", "E_D", "E_L", "E_R", "E_U"]
HEADS = ["U", "D", "L", "R"]
for h in HEADS:
    names += [f"r_{h}", f"T_{h}", f"e_{h}"]
IX = {n: i for i, n in enumerate(names)}
D = len(names)

BETA = 40.0            # position-match sharpness: wrong keys are at least e^-40 down
GAMMA = BETA * 10000   # newline bonus for off-grid targets; beats any position score (<= beta * 2 * 31^2)
M = 10.0 * R_LOOPS     # gating constant, larger than any |T difference|

# ---------------- embeddings ----------------
E0 = np.zeros((16, D))   # low nibble of the code point
E1 = np.zeros((16, D))   # next nibble
E2 = np.zeros((32, D))   # position digit lo
E3 = np.zeros((32, D))   # position digit hi
E4 = np.zeros((2, D))    # first token of the text
E4[1, IX["invL"]] = 1.0  # p = 0: nothing to the left
E1[:, IX["one"]] = 1.0
# ' ' = 0x20, '#' = 0x23, 'S' = 0x53, 'G' = 0x47, '\n' = 0x0A
E1[0, IX["nl"]] = 1.0
E1[2, IX["space"]] = 1.0
E0[3, IX["space"]] -= 1.0
E0[3, IX["wall"]] += 1.0
E1[5, IX["wall"]] -= 1.0
E1[5, IX["space"]] += 1.0
E1[5, IX["S"]] = 1.0
E1[5, IX["P"]] = 1.0       # the path starts at S
E1[4, IX["G"]] = 1.0
E1[4, IX["r"]] = 1.0       # the BFS starts at G
for a in range(32):
    E2[a, IX["lo"]] = a
    E2[a, IX["sq"]] = a * a
    E2[a, IX["Lm"]] = (a - 1) % 32
    E2[a, IX["Lp"]] = (a + 1) % 32
    E2[a, IX["c0"]] = float(a == 0)
    E2[a, IX["c31"]] = float(a == 31)
    if a in (0, 1):
        E2[a, IX["invD"]] = -1.0
    if a == 31:
        E2[a, IX["invU"]] = -1.0      # p = 31 is row 1: it has a row above
for b in range(32):
    E3[b, IX["hi"]] = b
    E3[b, IX["sq"]] = b * b
    E3[b, IX["invU"]] = float(b == 0)                       # p <= 30: no row above (with the lo correction)
    E3[b, IX["invD"]] += {28: 1.0, 29: 2.0}.get(b, 0.0)     # p >= 898: no row below (with the lo correction)
    E3[b, IX["invR"]] = float(b == 29)                      # p = 928: last cell, nothing to the right

# ---------------- layer 0: attention (4 neighbour heads) + MLP1 ----------------
DK, DV = 4, 3
Wq = np.zeros((D, DK * len(HEADS)))
Wk = np.zeros((D, DK * len(HEADS)))
Wv = np.zeros((D, DV * len(HEADS)))
Wo = np.zeros((DV * len(HEADS), D))
s = np.sqrt(DK)                      # the runner divides q by sqrt(dk)
spec = {  # target lo column, target hi as {dim: coef}, edge flag, exported E read from that neighbour
    "U": ("Lp", {"hi": 1, "one": -1, "c31": 1}, "invU", "E_D"),
    "D": ("Lm", {"hi": 1, "one": 1, "c0": -1}, "invD", "E_U"),
    "L": ("Lm", {"hi": 1, "c0": -1}, "invL", "E_R"),
    "R": ("Lp", {"hi": 1, "c31": 1}, "invR", "E_L"),
}
for j, h in enumerate(HEADS):
    lt, ht, inv, ex = spec[h]
    q0 = DK * j
    Wq[IX[lt], q0 + 0] = 2 * BETA * s
    for n, c in ht.items():
        Wq[IX[n], q0 + 1] = 2 * BETA * s * c
    Wq[IX["one"], q0 + 2] = -BETA * s
    if inv:
        Wq[IX[inv], q0 + 3] = GAMMA * s
    Wk[IX["lo"], q0 + 0] = 1
    Wk[IX["hi"], q0 + 1] = 1
    Wk[IX["sq"], q0 + 2] = 1
    Wk[IX["nl"], q0 + 3] = 1
    v0 = DV * j
    for t, (src, dst) in enumerate([("r", f"r_{h}"), ("T", f"T_{h}"), (ex, f"e_{h}")]):
        Wv[IX[src], v0 + t] = 1
        Wo[v0 + t, IX[dst]] = 1

units = []   # each: (dict of input coefs, bias, dict of output coefs)


def unit(inp, bias, out):
    units.append((inp, bias, out))


def clamp_into(inp, bias, out_dim):
    """out_dim += clamp(inp + bias, 0, 1) via relu(a) - relu(a - 1)."""
    unit(inp, bias, {out_dim: 1.0})
    unit(inp, bias - 1.0, {out_dim: -1.0})


openc = {"space": M, "S": M, "G": M}
# reached: an unreached open cell with a reached neighbour becomes reached
inp = {f"r_{h}": 2.0 for h in HEADS}
inp.update({"r": -M, **openc})
clamp_into(inp, -0.5 - M, "r")
# T += 1 - r (old r)
unit({"r": 1.0}, 0.0, {"T": -1.0})
# pointer candidates, set once when the cell is reached at the start of the loop
for dname, h in [("C_D", "D"), ("C_L", "L"), ("C_R", "R"), ("C_U", "U")]:
    clamp_into({"T": 2.0, f"T_{h}": -2.0, "r": M, dname: -M}, -0.5 - M, dname)
# on path: a neighbour on the path points here
clamp_into({**{f"e_{h}": 2.0 for h in HEADS}, "P": -10.0}, -0.5, "P")
# clear the scratch dims the heads wrote
for h in HEADS:
    for nm in (f"r_{h}", f"T_{h}", f"e_{h}"):
        unit({nm: 1.0}, 0.0, {nm: -1.0})


def mlp(us):
    W1 = np.zeros((D, len(us)))
    b1 = np.zeros(len(us))
    W2 = np.zeros((len(us), D))
    for u, (inp, bias, out) in enumerate(us):
        for n, c in inp.items():
            W1[IX[n], u] += c
        b1[u] = bias
        for n, c in out.items():
            W2[u, IX[n]] += c
    return W1, b1, W2


W1a, b1a, W2a = mlp(units)
b2a = np.zeros(D)
b2a[IX["T"]] = 1.0

# ---------------- layer 1: MLP only: export the chosen direction of path cells ----------------
units = []
order = ["C_D", "C_L", "C_R", "C_U"]
for k, (cn, en) in enumerate(zip(order, ["E_D", "E_L", "E_R", "E_U"])):
    inp = {"P": 1.0, cn: 1.0, en: -2.0}
    for prev in order[:k]:
        inp[prev] = -1.0
    unit(inp, -1.0, {en: 1.0})
W1b, b1b, W2b = mlp(units)
b2b = np.zeros(D)

# ---------------- readout ----------------
alphabet = ["", "#", "\n", "S", "G", " ", "o"]
Wout = np.zeros((D, len(alphabet)))
bout = np.zeros(len(alphabet))
bout[0] = -100.0
Wout[IX["wall"], 1] = 10
Wout[IX["nl"], 2] = 10
Wout[IX["S"], 3] = 10
Wout[IX["G"], 4] = 10
Wout[IX["space"], 5] = 1
Wout[IX["P"], 6] = 2
Wout[IX["S"], 6] = -5
Wout[IX["G"], 6] = -5

os.makedirs(OUT, exist_ok=True)
arrays = {"emb.0": E0, "emb.1": E1, "emb.2": E2, "emb.3": E3, "emb.4": E4,
          "L0.Wq": Wq, "L0.Wk": Wk, "L0.Wv": Wv, "L0.Wo": Wo,
          "L0.W1": W1a, "L0.b1": b1a, "L0.W2": W2a, "L0.b2": b2a,
          "L1.W1": W1b, "L1.b1": b1b, "L1.W2": W2b, "L1.b2": b2b,
          "out.W": Wout, "out.b": bout}
np.savez(os.path.join(OUT, "weights.npz"), **arrays)
cfg = {"d_model": D,
       "fields": ["char_nibble0", "char_nibble1", "pos_digit0", "pos_digit1", "first"],
       "layers": [{"heads": [{"dk": DK, "dv": DV, "mask": "none"} for _ in HEADS], "d_ff": W1a.shape[1]},
                  {"heads": [], "d_ff": W1b.shape[1]}],
       "loops": {"const": R_LOOPS, "lin": 0, "div": 1, "log2": 0},
       "alphabet": alphabet}
json.dump(cfg, open(os.path.join(OUT, "network.json"), "w"), indent=1)
print("d_model", D, "d_ff", W1a.shape[1], W1b.shape[1], "weights", sum(a.size for a in arrays.values()))
