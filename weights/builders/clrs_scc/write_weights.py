"""Writes every weight of the looped SCC transformer. Run: python3 -I work/write_weights.py

Character-level tokens (char code, position). One shared block of 7 layers, looped R times. Every state dimension
is "set" each loop (the MLP adds new - old), attention results land in scratch dims that the following MLP clears.
All boolean quantities go through flat steps  step(z) = relu(3z-1) - relu(3z-2)  (0 for z<=1/3, 1 for z>=2/3), so
float errors never accumulate across loops.

Per loop:
 L1 attn: position of the first '[' (outer), first ']' (end of row 0), last '[' at or before each token.
    MLP : n one-hot, q = offset of the token's row bracket from row 0's, column one-hot / '1'-bit at column.
 L2 MLP : row index one-hot (q = u * (2n+3)), '1'-bit at row, in-matrix flag, valid-node mask.
 L3 attn: every token of row u gathers row u of A and column u of A.  MLP: binarise, node / first / last flags.
 L4 attn: reachability squaring forward (F) and backward (B).  MLP: F, B, AU = A[u] and not visited.
 L5 attn: read AU[cur], parent[cur].  MLP: one DFS step (visit lowest unvisited child, else backtrack, else restart).
 L6 MLP : the newly visited node records its parent and discovery index; S = F and B (the SCC).
 L7 attn: per node, the SCC member with least discovery index.  MLP: output indicators.
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "net")
N = 24                       # most nodes handled
CHARS, POSN = 128, 4096
LAM = 50.0                   # attention sharpness (score gap)
C_BIG = 5000.0               # > any position
MG = 1e4                     # gate constant for integer comparisons on positions


class Dims:
    def __init__(self):
        self.n = 0

    def __call__(self, k=1):
        a = self.n
        self.n += k
        return a if k == 1 else list(range(a, a + k))


D = Dims()
ONE, P, ISO, ISC, IS1 = D(), D(), D(), D(), D()
sPO, sFC, sRB = D(), D(), D()
Q = D()
ON = D(N + 1)            # one-hot of n (number of nodes), index 0..N
VAL = D(N)               # [k < n]
OJ = D(N)                # column one-hot (offset in row = 2j)
BR = D(N)                # '1' at column j
OI = D(N)                # row one-hot
BC = D(N)                # '1' at row i
INM = D()
gAr, gAc = D(N), D(N)
AR, AC = D(N), D(N)
ISN, ISF, ISL = D(), D(), D()
gF, gB = D(N), D(N)
F, B, AU = D(N), D(N), D(N)
gAU, gPar = D(N), D(N)
CUR, PREV, V = D(N), D(N), D(N)
T = D()
PAR = D(N)
DISC, REC = D(), D()
S = D(N)
gId = D(N)
Om, Of, Ol = D(N), D(N), D(N)
DM = D.n


class MLP:
    def __init__(self):
        self.units = []      # (inw dict, bias, outw dict)

    def unit(self, inw, b, outw):
        self.units.append((dict(inw), float(b), dict(outw)))

    def step(self, inw, b, outw):
        """adds outw * step(z), z = inw.h + b"""
        self.unit({k: 3 * v for k, v in inw.items()}, 3 * b - 1, outw)
        self.unit({k: 3 * v for k, v in inw.items()}, 3 * b - 2, {k: -v for k, v in outw.items()})

    def lin(self, inw, b, outw):
        """adds outw * (z) for a z of any sign"""
        self.unit(inw, b, outw)
        self.unit({k: -v for k, v in inw.items()}, -b, {k: -v for k, v in outw.items()})

    def clear(self, dims):
        """subtract the old value of nonnegative dims"""
        for d in dims:
            self.unit({d: 1.0}, 0.0, {d: -1.0})

    def arrays(self):
        H = max(1, len(self.units))
        W1, b1, W2 = np.zeros((DM, H)), np.zeros(H), np.zeros((H, DM))
        for j, (iw, b, ow) in enumerate(self.units):
            for k, v in iw.items():
                W1[k, j] += v
            b1[j] = b
            for k, v in ow.items():
                W2[j, k] += v
        return W1, b1, W2, np.zeros(DM)


class Head:
    def __init__(self, dk, dv, mask):
        self.dk, self.dv, self.mask = dk, dv, mask
        self.Wq, self.Wk = np.zeros((DM, dk)), np.zeros((DM, dk))
        self.Wv, self.Wo = np.zeros((DM, dv)), np.zeros((dv, DM))

    def score(self, c, qw, kw):
        """adds (qw.h_q) * (kw.h_k) to the score in key/query channel c (pre-scaled for 1/sqrt(dk))"""
        for k, v in qw.items():
            self.Wq[k, c] += v * np.sqrt(self.dk)
        for k, v in kw.items():
            self.Wk[k, c] += v

    def move(self, src, dst):
        for c, (s, d) in enumerate(zip(src, dst)):
            self.Wv[s, c] = 1.0
            self.Wo[c, d] = 1.0


def match_head(qdims, kdims, vsrc, vdst, extra=None, lam=LAM):
    h = Head(len(qdims) + (1 if extra else 0), len(vsrc), "none")
    for c, (a, b) in enumerate(zip(qdims, kdims)):
        h.score(c, {a: lam}, {b: 1.0})
    if extra:
        h.score(len(qdims), extra[0], extra[1])
    h.move(vsrc, vdst)
    return h


layers = []   # (heads, mlp)

# ---------------- L1 ----------------
h1 = Head(1, 1, "none"); h1.score(0, {ONE: LAM}, {ISO: C_BIG, P: -1.0}); h1.move([P], [sPO])
h2 = Head(1, 1, "none"); h2.score(0, {ONE: LAM}, {ISC: C_BIG, P: -1.0}); h2.move([P], [sFC])
h3 = Head(1, 1, "causal"); h3.score(0, {ONE: LAM}, {ISO: C_BIG, P: 1.0}); h3.move([P], [sRB])
m = MLP()
# n = (sFC - sPO - 1 - 1) / 2   (row 0 bracket at sPO + 1, its ']' at sPO + 1 + 2n)
nz = {sFC: 0.5, sPO: -0.5}
nb = -0.5
for k in range(N + 1):     # [n == k] = step(n - k + 1) - step(n - k)
    m.step(nz, nb - k + 1, {ON[k]: 1.0})
    m.step(nz, nb - k, {ON[k]: -1.0})
m.clear(ON)
m.lin({sRB: 1.0, sPO: -1.0}, -1.0, {Q: 1.0})       # q = sRB - (sPO + 1)
m.lin({Q: 1.0}, 0.0, {Q: -1.0})
# o = P - sRB - 1 ; column j when o == 2j
for j in range(N):
    z = {P: 1.0, sRB: -1.0}
    m.step(z, -1.0 - 2 * j + 1, {OJ[j]: 1.0})
    m.step(z, -1.0 - 2 * j, {OJ[j]: -1.0})
    zg = dict(z); zg[IS1] = MG
    m.step(zg, -1.0 - 2 * j + 1 - MG, {BR[j]: 1.0})
    m.step(zg, -1.0 - 2 * j - MG, {BR[j]: -1.0})
m.clear(OJ + BR + [sPO, sFC, sRB])
layers.append(([h1, h2, h3], m))

# ---------------- L2 ----------------
m = MLP()
for k in range(1, N + 1):
    s = 2 * k + 3
    for u in range(k):
        # [q - u s == 0] gated by ON[k]
        zg = {Q: 1.0, ON[k]: MG}
        m.step(zg, -u * s + 1 - MG, {OI[u]: 1.0, INM: 1.0})
        m.step(zg, -u * s - MG, {OI[u]: -1.0, INM: -1.0})
        zg2 = {Q: 1.0, ON[k]: MG, IS1: MG}
        m.step(zg2, -u * s + 1 - 2 * MG, {BC[u]: 1.0})
        m.step(zg2, -u * s - 2 * MG, {BC[u]: -1.0})
for mm in range(N + 1):
    m.unit({ON[mm]: 1.0}, 0.0, {VAL[k]: 1.0 for k in range(min(mm, N))})
m.clear(OI + BC + VAL + [INM])
layers.append(([], m))

# ---------------- L3 ----------------
hr = match_head(OI, OI, BR, gAr)
hc = match_head(OI, OJ, BC, gAc, extra=({ONE: LAM}, {INM: 1.0}))
m = MLP()
for j in range(N):
    m.step({gAr[j]: 100.0}, 0.0, {AR[j]: 1.0})
    m.step({gAc[j]: 100.0}, 0.0, {AC[j]: 1.0})
m.step({ISO: 1.0, **{OI[u]: 1.0 for u in range(N)}}, -1.0, {ISN: 1.0})
m.step({ISO: 1.0, OI[0]: 1.0}, -1.0, {ISF: 1.0})
for u in range(N):
    m.step({ISO: 1.0, OI[u]: 1.0, ON[u + 1]: 1.0}, -2.0, {ISL: 1.0})
m.clear(gAr + gAc + AR + AC + [ISN, ISF, ISL])
layers.append(([hr, hc], m))

# ---------------- L4 ----------------
hf = match_head(F, OI, F, gF)
hb = match_head(B, OI, B, gB)
m = MLP()
for j in range(N):
    m.step({AR[j]: 1.0, OI[j]: 1.0, gF[j]: 40.0}, 0.0, {F[j]: 1.0})
    m.step({AC[j]: 1.0, OI[j]: 1.0, gB[j]: 40.0}, 0.0, {B[j]: 1.0})
    m.step({AR[j]: 1.0, V[j]: -1.0}, 0.0, {AU[j]: 1.0})
m.clear(gF + gB + F + B + AU)
layers.append(([hf, hb], m))

# ---------------- L5: DFS step ----------------
hd = Head(N, 2 * N, "none")
for c in range(N):
    hd.score(c, {CUR[c]: LAM}, {OI[c]: 1.0})
hd.move(AU + PAR, gAU + gPar)
m = MLP()
has = {CUR[c]: 1.0 for c in range(N)}
for k in range(N):
    first_out = {CUR[k]: 1.0, V[k]: 1.0, T: 1.0}
    z1 = {gAU[k]: 1.0}
    for j in range(k):
        z1[gAU[j]] = -1.0
    for c in range(N):
        z1[CUR[c]] = z1.get(CUR[c], 0.0) + 3.0
    m.step(z1, -3.0, first_out)                       # child of cur
    z2 = {VAL[k]: 1.0, V[k]: -1.0}
    for j in range(k):
        z2[VAL[j]] = z2.get(VAL[j], 0.0) - 1.0
        z2[V[j]] = z2.get(V[j], 0.0) + 1.0
    for c in range(N):
        z2[CUR[c]] = z2.get(CUR[c], 0.0) - 3.0
    m.step(z2, 0.0, first_out)                        # restart at lowest unvisited
    z3 = {gPar[k]: 1.0}
    for c in range(N):
        z3[CUR[c]] = 3.0
        z3[gAU[c]] = -3.0
    m.step(z3, -3.0, {CUR[k]: 1.0})                   # backtrack
    m.unit({CUR[k]: 1.0}, 0.0, {CUR[k]: -1.0, PREV[k]: 1.0})
m.clear(PREV + gAU + gPar)
layers.append(([hd], m))

# ---------------- L6: record visit, SCC sets ----------------
m = MLP()
for j in range(N):
    for k in range(N):
        m.step({PREV[k]: 1.0, CUR[j]: 1.0, OI[j]: 1.0, REC: -1.0}, -2.0, {PAR[k]: 1.0})
    m.unit({T: 1.0, CUR[j]: 100.0, OI[j]: 100.0, REC: -100.0}, -200.0, {DISC: 1.0})
    m.step({CUR[j]: 1.0, OI[j]: 1.0, REC: -1.0}, -1.0, {REC: 1.0})
    m.step({F[j]: 1.0, B[j]: 1.0}, -1.0, {S[j]: 1.0})
m.clear(S)
layers.append(([], m))

# ---------------- L7: SCC id = member with least discovery index ----------------
MU = 40.0
hs = Head(N + 1, N, "none")
for c in range(N):
    hs.score(c, {S[c]: MU * (N + 2) + 40.0}, {OI[c]: 1.0})
hs.score(N, {ONE: -MU}, {DISC: 1.0})
hs.move(OI, gId)
m = MLP()
for k in range(N):
    m.step({gId[k]: 1.0, ISN: 1.0, ISF: -1.0, ISL: -1.0}, -1.0, {Om[k]: 1.0})
    m.step({gId[k]: 1.0, ISF: 1.0}, -1.0, {Of[k]: 1.0})
    m.step({gId[k]: 1.0, ISL: 1.0}, -1.0, {Ol[k]: 1.0})
m.clear(gId + Om + Of + Ol)
layers.append(([hs], m))

# ---------------- embedding, readout ----------------
E0 = np.zeros((CHARS, DM))
E0[ord("["), ISO] = 1.0
E0[ord("]"), ISC] = 1.0
E0[ord("1"), IS1] = 1.0
E1 = np.zeros((POSN, DM))
E1[:, ONE] = 1.0
E1[:, P] = np.arange(POSN)

alphabet = [""]
cols = {}
for k in range(N):
    for dims, fmt in ((Om, f"{k} "), (Of, f"[{k} "), (Ol, f"{k}]")):
        cols[len(alphabet)] = dims[k]
        alphabet.append(fmt)
Wout = np.zeros((DM, len(alphabet)))
bout = np.zeros(len(alphabet))
bout[0] = 0.5
for a, d in cols.items():
    Wout[d, a] = 1.0

# ---------------- write ----------------
os.makedirs(OUT, exist_ok=True)
W = {"emb.0": E0, "emb.1": E1, "out.W": Wout, "out.b": bout}
cfg_layers = []
for l, (heads, mlp) in enumerate(layers):
    if heads:
        W[f"L{l}.Wq"] = np.hstack([h.Wq for h in heads])
        W[f"L{l}.Wk"] = np.hstack([h.Wk for h in heads])
        W[f"L{l}.Wv"] = np.hstack([h.Wv for h in heads])
        W[f"L{l}.Wo"] = np.vstack([h.Wo for h in heads])
    W1, b1, W2, b2 = mlp.arrays()
    W[f"L{l}.W1"], W[f"L{l}.b1"], W[f"L{l}.W2"], W[f"L{l}.b2"] = W1, b1, W2, b2
    cfg_layers.append({"heads": [{"dk": h.dk, "dv": h.dv, "mask": h.mask} for h in heads], "d_ff": W1.shape[1]})

store = {}
for k, a in W.items():
    if a.size > 4096 and np.count_nonzero(a) < a.size / 4:
        idx = np.flatnonzero(a)
        store[k + ".shape"] = np.array(a.shape, dtype=np.int64)
        store[k + ".idx"] = idx.astype(np.int64)
        store[k + ".val"] = a.ravel()[idx]
    else:
        store[k] = a
np.savez_compressed(os.path.join(OUT, "weights.npz"), **store)
cfg = {"d_model": DM, "fields": ["char", "position"], "layers": cfg_layers,
       "loops": {"const": 12, "lin": 1, "div": 16, "log2": 0}, "alphabet": alphabet}
json.dump(cfg, open(os.path.join(OUT, "network.json"), "w"), indent=1)
tot = sum(a.size for a in W.values())
nnz = sum(int(np.count_nonzero(a)) for a in W.values())
print(f"d_model {DM}, d_ff {[c['d_ff'] for c in cfg_layers]}, params {tot}, nonzero {nnz}")
