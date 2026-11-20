"""Writes every weight of the looped Sudoku network (nothing is trained or fitted).

Run:  python3 -I work/write_weights.py        -> work/net/{network.json, weights.npz}  (tokenizer.py is hand written)

What the network computes
-------------------------
Each of the 81 cell tokens holds its candidate set A (9 bits), a stack of D saved candidate sets S[0..D-1] and the
current depth as a one-hot Lv[0..D] (the same in every token). One "step" (5 layers) does:

  a  MLP : soh = A if the cell has exactly one candidate; emp = no candidate; u = two or more candidates
  b  attn: per-digit candidate counts in the cell's row / column / box; per-digit "a peer is solved with it"
     MLP : elim (candidate taken by a solved peer), conf (cell's own value taken by a peer), hs (hidden-single
           flags), zsum (some digit has no place in one of the cell's units), p (MRV priority, 0 if solved)
  c  MLP : NA = propagated candidate set (naked + hidden singles)
  d  attn: global OR of contradiction / change / unsolved over the 81 cells; max of p (MRV cell choice)
     MLP : POP, CH, UN flags; ch = this cell is the MRV cell; fd = its lowest candidate
  e  MLP : the update
           contradiction           -> A := S[L-1], L := L-1                            (backtrack)
           else change             -> A := NA                                          (propagate)
           else unsolved           -> S[L] := A minus (cell ch, digit fd); cell ch := fd; L := L+1   (branch)
           else                    -> nothing (solved; the state is frozen)
           and every temporary is cleared exactly.

The block is K copies of the step; it runs R = 1000 times. The readout writes each cell's digit and a newline at the
newline tokens between grid rows.
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "net")

D = 24          # stack depth
K = 12          # steps per loop
R = 1000        # loops
BIG_ATT = 30.0  # attention score margin

PREFIX = 19     # len("Solve this Sudoku:\n"); cell (r, c) is at character 19 + 10 r + c


def cell_pos(i):
    return PREFIX + 10 * (i // 9) + (i % 9)


# ---------------------------------------------------------------- residual layout
class Layout:
    def __init__(self):
        self.n = 0
        self.names = {}

    def add(self, name, size=1):
        idx = list(range(self.n, self.n + size))
        self.n += size
        self.names[name] = idx
        return idx if size > 1 else idx[0]


Lt = Layout()
ONE = Lt.add("one"); ISCELL = Lt.add("iscell"); ISNL = Lt.add("isnl"); TIE = Lt.add("tie")
CELL = Lt.add("cellOH", 81); ROWOH = Lt.add("rowOH", 9); COLOH = Lt.add("colOH", 9); BOXOH = Lt.add("boxOH", 9)
A = Lt.add("A", 9)
S = [Lt.add(f"S{m}", 9) for m in range(D)]
LV = Lt.add("Lv", D + 1)
TEMPS = []


def temp(name, size=1):
    r = Lt.add(name, size)
    TEMPS.extend(r if size > 1 else [r])
    return r


SOH = temp("soh", 9); EMP = temp("emp"); U = temp("u")
CR = temp("cr", 9); CC = temp("cc", 9); CB = temp("cb", 9); PS = temp("ps", 9)
ELIM = temp("elim", 9); CONF = temp("conf"); HS = temp("hs", 9); ZSUM = temp("zsum"); P = temp("p")
NA = temp("na", 9)
GC = temp("gc"); GCH = temp("gch"); GU = temp("gu"); PMAX = temp("pmax")
POPF = temp("popf"); CHF = temp("chf"); UNF = temp("unf"); CH = temp("ch"); FD = temp("fd", 9)
DM = Lt.n


# ---------------------------------------------------------------- layer builder
class Layer:
    def __init__(self):
        self.units = []   # (in: {dim: coef}, bias, out: {dim: coef})
        self.heads = []   # (dk, dv, Wq (DM,dk), Wk (DM,dk), Wv (DM,dv), Wo (dv,DM))

    def relu(self, inp, bias, out):
        self.units.append((dict(inp), float(bias), dict(out)))

    def step(self, inp, bias, out_dim, coef=1.0):
        """adds coef * [X >= 1/2] to out_dim, X = inp.h + bias, exact for X <= 1/4 and X >= 3/4."""
        self.relu(inp, bias - 0.25, {out_dim: 2.0 * coef})
        self.relu(inp, bias - 0.75, {out_dim: -2.0 * coef})

    def conj(self, lits, out_dim, coef=1.0):
        """coef * AND of binary literals; lits = list of (dim, positive?)."""
        inp, bias = {}, 0.0
        for dim, pos in lits:
            if pos:
                inp[dim] = inp.get(dim, 0.0) + 1.0
            else:
                inp[dim] = inp.get(dim, 0.0) - 1.0
                bias += 1.0
        self.step(inp, bias - (len(lits) - 1), out_dim, coef)

    def clear(self, dim):
        self.relu({dim: 1.0}, 0.0, {dim: -1.0})
        self.relu({dim: -1.0}, 0.0, {dim: 1.0})

    def head(self, dk, dv, q, k, v):
        """q, k: (DM, dk) arrays, already including the 1/sqrt(dk) compensation; v: list of (dim->coef dict, out dim)."""
        Wv = np.zeros((DM, dv)); Wo = np.zeros((dv, DM))
        for j, (vin, odim) in enumerate(v):
            for dmn, c in vin.items():
                Wv[dmn, j] += c
            Wo[j, odim] = 1.0
        self.heads.append((dk, dv, q, k, Wv, Wo))

    def arrays(self):
        ff = len(self.units)
        W1 = np.zeros((DM, ff)); b1 = np.zeros(ff); W2 = np.zeros((ff, DM))
        for j, (inp, b, out) in enumerate(self.units):
            for dmn, c in inp.items():
                W1[dmn, j] += c
            b1[j] = b
            for dmn, c in out.items():
                W2[j, dmn] += c
        res = {"W1": W1, "b1": b1, "W2": W2, "b2": np.zeros(DM)}
        if self.heads:
            res["Wq"] = np.hstack([h[2] for h in self.heads])
            res["Wk"] = np.hstack([h[3] for h in self.heads])
            res["Wv"] = np.hstack([h[4] for h in self.heads])
            res["Wo"] = np.vstack([h[5] for h in self.heads])
        return res


ROWI = np.arange(81) // 9; COLI = np.arange(81) % 9; BOXI = (ROWI // 3) * 3 + COLI // 3
PEER = ((ROWI[:, None] == ROWI[None]) | (COLI[:, None] == COLI[None]) | (BOXI[:, None] == BOXI[None])) \
    & ~np.eye(81, dtype=bool)


def unit_head(L, ohdims, outdims):
    q = np.zeros((DM, 9)); k = np.zeros((DM, 9))
    for j in range(9):
        q[ohdims[j], j] = BIG_ATT * 3.0      # sqrt(9) = 3
        k[ohdims[j], j] = 1.0
    L.head(9, 9, q, k, [({A[d]: 1.0}, outdims[d]) for d in range(9)])


def build_step():
    layers = []
    cnt = {A[d]: 1.0 for d in range(9)}

    # a: soh, emp, u
    La = Layer()
    for d in range(9):
        La.step({**{A[e]: -1.0 for e in range(9)}, A[d]: 1.0}, 0.0, SOH[d])     # 2A_d - cnt
    La.step({A[d]: -1.0 for d in range(9)}, 1.0, EMP)
    La.step(cnt, -1.0, U)
    layers.append(La)

    # b: unit counts and peer-solved; then elim, conf, hs, zsum, p
    Lb = Layer()
    unit_head(Lb, ROWOH, CR); unit_head(Lb, COLOH, CC); unit_head(Lb, BOXOH, CB)
    q = np.zeros((DM, 81)); k = np.zeros((DM, 81))
    for i in range(81):
        q[CELL[i], i] = BIG_ATT * 9.0        # sqrt(81)
        for j in range(81):
            if PEER[i, j]:
                k[CELL[j], i] = 1.0
    Lb.head(81, 9, q, k, [({SOH[d]: 1.0}, PS[d]) for d in range(9)])
    for d in range(9):
        Lb.step({PS[d]: 20.0, A[d]: 20.0}, -20.0, ELIM[d])
        Lb.step({PS[d]: 20.0, SOH[d]: 20.0}, -20.0, CONF)
        for cu in (CR, CC, CB):
            Lb.step({A[d]: 2.0, cu[d]: -9.0}, 0.0, HS[d])
            Lb.step({cu[d]: -9.0}, 1.0, ZSUM)
    Lb.relu({**{A[d]: -100.0 for d in range(9)}, TIE: 1.0, U: 1000.0}, -100.0, {P: 1.0})
    layers.append(Lb)

    # c: NA
    Lc = Layer()
    for d in range(9):
        inp = {A[d]: 1.0, ELIM[d]: -1.0}
        for e in range(9):
            if e != d:
                inp[HS[e]] = -1.0
        Lc.step(inp, 0.0, NA[d])
    layers.append(Lc)

    # d: global flags, MRV
    Ld = Layer()
    q = np.zeros((DM, 1)); k = np.zeros((DM, 1))
    q[ONE, 0] = BIG_ATT; k[ISCELL, 0] = 1.0
    vc = {EMP: 1.0, CONF: 1.0, ZSUM: 1.0}
    vch = {**{A[d]: 1.0 for d in range(9)}, **{NA[d]: -1.0 for d in range(9)}}
    Ld.head(1, 3, q, k, [(vc, GC), (vch, GCH), ({U: 1.0}, GU)])
    q = np.zeros((DM, 1)); k = np.zeros((DM, 1))
    q[ONE, 0] = BIG_ATT; k[P, 0] = 1.0
    Ld.head(1, 1, q, k, [({P: 1.0}, PMAX)])
    Ld.step({GC: 81.0}, 0.0, POPF)
    Ld.step({GCH: 81.0}, 0.0, CHF)
    Ld.step({GU: 81.0}, 0.0, UNF)
    Ld.step({P: 1.0, PMAX: -1.0}, 1.0, CH)
    for d in range(9):
        Ld.step({A[d]: 1.0, **{A[e]: -1.0 for e in range(d)}}, 0.0, FD[d])
    layers.append(Ld)

    # e: update
    Le = Layer()
    for d in range(9):
        a = A[d]
        # -A
        Le.clear(a)
        # restore: POP & Lv[m] & S[m-1]_d
        for m in range(1, D + 1):
            Le.conj([(POPF, True), (LV[m], True), (S[m - 1][d], True)], a)
        # propagate: CH & !POP & NA_d
        Le.conj([(CHF, True), (POPF, False), (NA[d], True)], a)
        # branch, chosen cell: UN & !CH & !POP & ch & fd_d
        Le.conj([(UNF, True), (CHF, False), (POPF, False), (CH, True), (FD[d], True)], a)
        # keep: A & !POP & !CH & !(UN & ch)
        Le.conj([(a, True), (POPF, False), (CHF, False)], a)
        Le.conj([(a, True), (POPF, False), (CHF, False), (UNF, True), (CH, True)], a, -1.0)
        # slots: on branch at level m, S[m] := A minus (ch & fd)
        for m in range(D):
            s = S[m][d]
            G = [(UNF, True), (CHF, False), (POPF, False), (LV[m], True)]
            Le.conj(G + [(a, True), (s, False)], s, 1.0)
            Le.conj(G + [(a, True), (CH, True), (FD[d], True), (s, False)], s, -1.0)
            Le.conj(G + [(s, True), (a, False)], s, -1.0)
            Le.conj(G + [(s, True), (CH, True), (FD[d], True)], s, -1.0)
    for m in range(D + 1):
        if m + 1 <= D:
            Le.conj([(POPF, True), (LV[m + 1], True)], LV[m])
        if m - 1 >= 0:
            Le.conj([(UNF, True), (CHF, False), (POPF, False), (LV[m - 1], True)], LV[m])
        Le.conj([(POPF, True), (LV[m], True)], LV[m], -1.0)
        Le.conj([(UNF, True), (CHF, False), (POPF, False), (LV[m], True)], LV[m], -1.0)
    for t in TEMPS:
        Le.clear(t)
    layers.append(Le)
    return layers


def embeddings():
    E0 = np.zeros((256, DM))
    E0[ord("0"), A] = 1.0
    for d in range(1, 10):
        E0[ord(str(d)), A[d - 1]] = 1.0
    E1 = np.zeros((1024, DM))
    E1[:, ONE] = 1.0
    E1[:, LV[0]] = 1.0
    for i in range(81):
        p = cell_pos(i)
        E1[p, ISCELL] = 1.0
        E1[p, TIE] = 81 - i
        E1[p, CELL[i]] = 1.0
        E1[p, ROWOH[ROWI[i]]] = 1.0
        E1[p, COLOH[COLI[i]]] = 1.0
        E1[p, BOXOH[BOXI[i]]] = 1.0
    for r in range(8):
        E1[PREFIX + 10 * r + 9, ISNL] = 1.0
    return E0, E1


ALPHABET = [""] + [str(d) for d in range(1, 10)] + ["\n"]


def readout():
    W = np.zeros((DM, len(ALPHABET))); b = np.zeros(len(ALPHABET))
    b[0] = 0.5
    for d in range(9):
        W[A[d], 1 + d] = 1.0
    W[ISNL, 10] = 2.0
    return W, b


def sparse(z, name, a):
    a = np.asarray(a, dtype=np.float64)
    nz = np.flatnonzero(a)
    if nz.size < a.size // 4:
        z[name + ".shape"] = np.array(a.shape, dtype=np.int64)
        z[name + ".idx"] = nz.astype(np.int64)
        z[name + ".val"] = a.ravel()[nz]
    else:
        z[name] = a


def main():
    step = build_step()
    block = step * K
    z = {}
    E0, E1 = embeddings()
    sparse(z, "emb.0", E0); sparse(z, "emb.1", E1)
    cfg_layers = []
    for l, L in enumerate(block):
        arr = L.arrays()
        for kname, v in arr.items():
            sparse(z, f"L{l}.{kname}", v)
        cfg_layers.append({"heads": [{"dk": h[0], "dv": h[1], "mask": "none"} for h in L.heads],
                           "d_ff": len(L.units)})
    W, b = readout()
    sparse(z, "out.W", W); z["out.b"] = b
    cfg = {"d_model": DM, "fields": ["byte", "position"], "layers": cfg_layers,
           "loops": {"const": R, "lin": 0, "div": 1, "log2": 0}, "alphabet": ALPHABET}
    os.makedirs(OUT, exist_ok=True)
    json.dump(cfg, open(os.path.join(OUT, "network.json"), "w"))
    np.savez_compressed(os.path.join(OUT, "weights.npz"), **z)
    print(f"d_model {DM}, layers/step {len(step)}, d_ff per step layer {[len(L.units) for L in step]}, K {K}, R {R}")


if __name__ == "__main__":
    main()
