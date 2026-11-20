"""Writes the hand-set looped transformer for Maze-Hard into work/net/.

    python3 -I work/write_weights.py

What the block computes, per loop (one layer: 4 attention heads + one MLP):
  * every token attends, by hard (large-margin) attention, to the token one line up / one line down / one column left /
    one column right of it (scores -beta * squared (line, column) distance to the target, built from the tokenizer's
    line and column fields).
  * wave from G: a cell becomes "visited" the loop after one of its 4 neighbours is; at that moment it latches a
    pointer to the first already-visited neighbour in the order Down, Left, Right, Up (the neighbours visited just
    before it are exactly those one step closer to G).
  * path from S: S, once its pointer is set, latches "send" bits s_dir = its pointer; a cell that receives from a
    neighbour whose send bit points at it does the same. The marked cells are the pointer chain S -> G, which is the
    lexicographically smallest (D<L<R<U) shortest path, i.e. the BFS-from-S parent path the dataset uses.
  * readout: 'o' where some send bit is set, except S and G; otherwise the input character.
"""
import json
import os
import shutil

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "net")

TOKENIZER = '''"""Generic character tokenizer. One token per character (every character of the text, unchanged), with fields:
  0..6  bit k of the character's code point (k = 0..6)
  7     1 if the code point is 128 or more, else 0
  8     line number of the character (newlines before it), mod 64
  9     column of the character (characters since the last newline), mod 64
Nothing here depends on what the text is about."""
import numpy as np

MOD = 64


def tokenize(text: str):
    out = []
    line = col = 0
    for ch in text:
        o = ord(ch)
        out.append([(o >> k) & 1 for k in range(7)] + [1 if o >= 128 else 0, line % MOD, col % MOD])
        if ch == "\\n":
            line, col = line + 1, 0
        else:
            col += 1
    return np.asarray(out, dtype=np.int64).reshape(len(out), 10)
'''

MOD = 64
BETA = 40.0          # attention sharpness per unit of squared grid distance
R_LOOPS = 480        # loops: wave G->S (d loops) then path S->G (d loops); enough for paths up to ~235 steps

# residual stream layout
names = ["one", "l", "c", "q2", "open", "SG", "vis",
         "pD", "pL", "pR", "pU", "sD", "sL", "sR", "sU",
         "nD", "nL", "nR", "nU", "recv"]
I = {k: i for i, k in enumerate(names)}
d = len(names)
DIRS = ["D", "L", "R", "U"]                       # pointer order = tie-break order
OFF = {"D": (1, 0), "L": (0, -1), "R": (0, 1), "U": (-1, 0)}
OPP = {"D": "U", "U": "D", "L": "R", "R": "L"}

alphabet = ["", " ", "#", "S", "G", "o", "\n"]

W = {}

# ---- embeddings: fields 0..7 bits (2 rows each), 8 line, 9 column ----
# per-character features as affine functions of the code-point bits (b0..b6):
#   open = b5 - b0 + 2 b6     (' ', 'S', 'G' -> 1;  '#', '\n' -> 0)
#   SG   = b4 - b2            ('S' -> 1, 'G' -> -1, else 0)
#   vis  = b2                 ('G' -> 1 at the start, else 0)
bitfeat = {0: {"open": -1}, 2: {"SG": -1, "vis": 1}, 4: {"SG": 1}, 5: {"open": 1}, 6: {"open": 2}}
for k in range(8):
    E = np.zeros((2, d))
    for f, v in bitfeat.get(k, {}).items():
        E[1, I[f]] = v
    if k == 0:
        E[:, I["one"]] = 1.0          # constant 1, whatever the bit
    W[f"emb.{k}"] = E
El = np.zeros((MOD, d))
Ec = np.zeros((MOD, d))
for v in range(MOD):
    El[v, I["l"]] = v
    El[v, I["q2"]] += v * v
    Ec[v, I["c"]] = v
    Ec[v, I["q2"]] += v * v
W["emb.8"], W["emb.9"] = El, Ec

# ---- attention: 4 heads, head h reads the neighbour at offset OFF[h] ----
DK, DV = 3, 2
nh = len(DIRS)
Wq = np.zeros((d, nh * DK))
Wk = np.zeros((d, nh * DK))
Wv = np.zeros((d, nh * DV))
Wo = np.zeros((nh * DV, d))
sq = np.sqrt(DK)
for h, dr in enumerate(DIRS):
    dl, dc = OFF[dr]
    q0 = h * DK
    # score = beta * (2 lt l' + 2 ct c' - (l'^2 + c'^2)) = -beta * |(l', c') - target|^2 + const(query)
    Wq[I["l"], q0 + 0] = 2 * BETA * sq
    Wq[I["one"], q0 + 0] = 2 * BETA * sq * dl
    Wq[I["c"], q0 + 1] = 2 * BETA * sq
    Wq[I["one"], q0 + 1] = 2 * BETA * sq * dc
    Wq[I["one"], q0 + 2] = -BETA * sq
    Wk[I["l"], q0 + 0] = 1
    Wk[I["c"], q0 + 1] = 1
    Wk[I["q2"], q0 + 2] = 1
    v0 = h * DV
    # value 0: the neighbour's visited bit -> n<dir>; value 1: the neighbour's send bit pointing back at me -> recv
    Wv[I["vis"], v0] = 1
    Wv[I["s" + OPP[dr]], v0 + 1] = 1
    Wo[v0, I["n" + dr]] = 1
    Wo[v0 + 1, I["recv"]] = 1
W["L0.Wq"], W["L0.Wk"], W["L0.Wv"], W["L0.Wo"] = Wq, Wk, Wv, Wo

# ---- MLP ----
units = []      # (input weights dict, bias, output weights dict)
# pointer latch (and visit): fires once, when an open unvisited cell first has a visited neighbour;
# the first visited neighbour in D, L, R, U order wins.
for j, dr in enumerate(DIRS):
    w = {"n" + dr: 1, "open": 4, "vis": -5}
    for e in DIRS[:j]:
        w["n" + e] = -1
    units.append((w, -4, {"p" + dr: 1, "vis": 1}))
# path latch: a cell not yet on the path that receives (or is S) copies its pointer into its send bits.
for dr in DIRS:
    w = {"p" + dr: 1, "recv": 1, "SG": 1}
    for e in DIRS:
        w["s" + e] = -3
    units.append((w, -1, {"s" + dr: 1}))
# clear the attention scratch (all values there are >= 0, so relu(x) = x exactly)
for k in ["nD", "nL", "nR", "nU", "recv"]:
    units.append(({k: 1}, 0, {k: -1}))
ff = len(units)
W1 = np.zeros((d, ff))
b1 = np.zeros(ff)
W2 = np.zeros((ff, d))
for u, (wi, b, wo) in enumerate(units):
    for k, v in wi.items():
        W1[I[k], u] = v
    b1[u] = b
    for k, v in wo.items():
        W2[u, I[k]] = v
W["L0.W1"], W["L0.b1"], W["L0.W2"], W["L0.b2"] = W1, b1, W2, np.zeros(d)

# ---- readout ----
Wout = np.zeros((d, len(alphabet)))
bout = np.zeros(len(alphabet))
A = {s: i for i, s in enumerate(alphabet)}
bout[A[""]] = -10
Wout[I["open"], A[" "]] = 1                      # open cell: ' ' = 1
bout[A["#"]] = 0.5
Wout[I["open"], A["#"]] = -1                     # wall: '#' = 0.5 beats ' ' = 0
Wout[I["c"], A["\n"]] = 1
bout[A["\n"]] = -29                              # column 30 (the newline): 1 beats '#' = 0.5
Wout[I["SG"], A["S"]] = 3
Wout[I["SG"], A["G"]] = -3
for dr in DIRS:
    Wout[I["s" + dr], A["o"]] = 2                # on the path: 'o' = 2 beats ' ' = 1 (S and G keep 3)
W["out.W"], W["out.b"] = Wout, bout

cfg = {"d_model": d,
       "fields": [f"bit{k}" for k in range(7)] + ["high", "line", "col"],
       "layers": [{"heads": [{"dk": DK, "dv": DV, "mask": "none"} for _ in DIRS], "d_ff": ff}],
       "loops": {"const": R_LOOPS, "lin": 0, "div": 1, "log2": 0},
       "alphabet": alphabet}

if os.path.isdir(OUT):
    shutil.rmtree(OUT)
os.makedirs(OUT)
json.dump(cfg, open(os.path.join(OUT, "network.json"), "w"), indent=1)
np.savez(os.path.join(OUT, "weights.npz"), **{k: np.asarray(v, dtype=np.float64) for k, v in W.items()})
open(os.path.join(OUT, "tokenizer.py"), "w").write(TOKENIZER)
print(f"wrote {OUT}: d_model {d}, d_ff {ff}, R {R_LOOPS}, {sum(v.size for v in W.values())} weights")
