"""Writes every weight of the looped maze solver (no training). Run: python3 -I work/write_weights.py

Residual stream (d = 25), per token (one token per character):
   0 one        constant 1                      (line table)
   1 r          line index                      (line table)
   2 c          column index                    (column table)
   3 q2         r^2 + c^2                       (line + column tables)
   4..8         is ' ', '#', '\\n', 'S', 'G'     (character table)
   9 rho        reached by the wave started at G (G starts at 1)
  10 age        loops since reached  =>  age(u) - age(v) = dist_G(v) - dist_G(u)
  11 pi         on the chosen path (S first, then one cell per loop)
  12..15 cX     neighbour in direction X (D, L, R, U) is one step closer to G (monotone flag)
  16 s          scratch: sum of the 4 neighbours' rho
  17..20        scratch: age of neighbour X
  21..24        scratch: "neighbour X points at me and is on the path" score

Block = 1 layer: 4 heads (one per direction X) each attend exactly to the neighbour cell in direction X (quadratic
position score with margin 2048, so softmax is an exact one-hot in float64; at a border the head finds the cell
itself, which is harmless). The MLP advances the wave, the ages, the closer-flags and the path, and clears scratch.
The path rule is the dataset's: from S, repeatedly step to the first neighbour, in order Down, Left, Right, Up, that
is one step closer to G (= the lexicographically smallest shortest path).
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "net")

D = 25
ONE, R_, C_, Q2 = 0, 1, 2, 3
SP, HS, NL, SS, GG = 4, 5, 6, 7, 8
RHO, AGE, PI = 9, 10, 11
CF = {"D": 12, "L": 13, "R": 14, "U": 15}
S_ = 16
AGEN = {"D": 17, "L": 18, "R": 19, "U": 20}
PIN = {"D": 21, "L": 22, "R": 23, "U": 24}
DIRS = ["D", "L", "R", "U"]                       # priority order of the dataset's path
OFF = {"D": (1, 0), "L": (0, -1), "R": (0, 1), "U": (-1, 0)}
OPP = {"D": "U", "U": "D", "L": "R", "R": "L"}
LAM = 4096.0                                      # score = -LAM/2 * squared grid distance (+ per-query constant)
K = float(2 ** 20)                                # latch constant for the closer-flags (ages stay far below)
DK = 4
R_RULE = {"const": 0, "lin": 1, "div": 2, "log2": 0}   # R = ceil(n / 2) = 450 for a 30x30 maze
ALPHABET = ["", " ", "#", "S", "G", "\n", "o"]


def main():
    W = {}
    # ---- embeddings
    E0 = np.zeros((128, D))
    for ch, dim in ((" ", SP), ("#", HS), ("\n", NL), ("S", SS), ("G", GG)):
        E0[ord(ch), dim] = 1
    E0[ord("G"), RHO] = 1                         # the wave starts at G
    E1 = np.zeros((32, D))
    E2 = np.zeros((32, D))
    for i in range(32):
        E1[i, ONE] = 1
        E1[i, R_] = i
        E1[i, Q2] += i * i
        E2[i, C_] = i
        E2[i, Q2] += i * i
    W["emb.0"], W["emb.1"], W["emb.2"] = E0, E1, E2

    # ---- attention: head h <-> direction X
    Wq = np.zeros((D, 4 * DK))
    Wk = np.zeros((D, 4 * DK))
    Wv = np.zeros((D, 4 * 3))
    Wo = np.zeros((4 * 3, D))
    for h, X in enumerate(DIRS):
        dr, dc = OFF[X]
        q0 = h * DK
        # query = LAM * (2(r+dr), 2(c+dc), -1, 0); key = (r', c', r'^2 + c'^2, 0)
        Wq[R_, q0 + 0] = 2 * LAM
        Wq[ONE, q0 + 0] = 2 * LAM * dr
        Wq[C_, q0 + 1] = 2 * LAM
        Wq[ONE, q0 + 1] = 2 * LAM * dc
        Wq[ONE, q0 + 2] = -LAM
        Wk[R_, q0 + 0] = 1
        Wk[C_, q0 + 1] = 1
        Wk[Q2, q0 + 2] = 1
        v0 = h * 3
        Wv[RHO, v0 + 0] = 1                       # neighbour's rho
        Wv[AGE, v0 + 1] = 1                       # neighbour's age
        # neighbour w in direction X points at me iff next(w) = OPP[X]; score = pi(w) + [next(w) = OPP[X] terms]
        Y = OPP[X]
        k = DIRS.index(Y)                         # number of higher-priority directions
        Wv[PI, v0 + 2] = 1
        Wv[CF[Y], v0 + 2] = 1
        for Z in DIRS[:k]:
            Wv[CF[Z], v0 + 2] = -1
        Wv[ONE, v0 + 2] = k                       # = 2 + k exactly when w is on the path and next(w) = Y
        Wo[v0 + 0, S_] = 1
        Wo[v0 + 1, AGEN[X]] = 1
        Wo[v0 + 2, PIN[X]] = 1
    W["L0.Wq"], W["L0.Wk"], W["L0.Wv"], W["L0.Wo"] = Wq, Wk, Wv, Wo

    # ---- MLP
    units = []                                    # (dict dim -> coef, bias, dict outdim -> coef)
    opn = {SP: 4, SS: 4, GG: 4}
    units.append(({S_: 1, RHO: -4, **opn}, -4, {RHO: 1}))      # wave: rho += min(1, s) if open and not reached
    units.append(({S_: 1, RHO: -4, **opn}, -5, {RHO: -1}))
    units.append(({RHO: 1}, 0, {AGE: 1}))                       # age += rho
    units.append(({SS: 1, RHO: 1, PI: -1}, -1, {PI: 1}))        # S joins the path once reached
    for X in DIRS:                                              # a path neighbour points at me
        k = DIRS.index(OPP[X])
        units.append(({PIN[X]: 1, PI: -2}, -(1 + k), {PI: 1}))
    for X in DIRS:                                              # closer flag, latched
        units.append(({AGEN[X]: 1, AGE: -1, CF[X]: -K}, 0, {CF[X]: 1}))
        units.append(({AGEN[X]: 1, AGE: -1, CF[X]: -K}, -1, {CF[X]: -1}))
    for dim in [S_] + [AGEN[X] for X in DIRS] + [PIN[X] for X in DIRS]:   # clear scratch (all values >= 0)
        units.append(({dim: 1}, 0, {dim: -1}))
    F = len(units)
    W1, b1, W2 = np.zeros((D, F)), np.zeros(F), np.zeros((F, D))
    for j, (inp, b, out) in enumerate(units):
        for i, v in inp.items():
            W1[i, j] = v
        b1[j] = b
        for i, v in out.items():
            W2[j, i] = v
    W["L0.W1"], W["L0.b1"], W["L0.W2"], W["L0.b2"] = W1, b1, W2, np.zeros(D)

    # ---- readout
    Wout, bout = np.zeros((D, len(ALPHABET))), np.zeros(len(ALPHABET))
    bout[ALPHABET.index("")] = 0.5
    Wout[SP, ALPHABET.index(" ")] = 1.5
    Wout[HS, ALPHABET.index("#")] = 3
    Wout[SS, ALPHABET.index("S")] = 3
    Wout[GG, ALPHABET.index("G")] = 3
    Wout[NL, ALPHABET.index("\n")] = 3
    Wout[SP, ALPHABET.index("o")] = 1
    Wout[PI, ALPHABET.index("o")] = 1
    W["out.W"], W["out.b"] = Wout, bout

    os.makedirs(OUT, exist_ok=True)
    np.savez(os.path.join(OUT, "weights.npz"), **W)
    cfg = {"d_model": D, "fields": ["char", "line", "column"],
           "layers": [{"heads": [{"dk": DK, "dv": 3, "mask": "none"} for _ in DIRS], "d_ff": F}],
           "loops": R_RULE, "alphabet": ALPHABET}
    json.dump(cfg, open(os.path.join(OUT, "network.json"), "w"), indent=1)
    print("wrote", OUT, "d_ff", F)


if __name__ == "__main__":
    main()
