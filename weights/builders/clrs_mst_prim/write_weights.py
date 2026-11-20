"""Writes every weight of the looped transformer for CLRS-Text mst_prim. Nothing is trained or fitted.

Run:  python3 -I work/write_weights.py      -> work/net/{network.json, weights.npz} (tokenizer.py is hand-written)

Block = 5 layers, run R = 2 + ceil(n/24) times.
  P1, P2 : parsing (idempotent "set" semantics, recomputed identically every loop)
  A      : each matrix entry (u,v) fetches age[u], age[v] from the node slots -> eligible(u,v)
  B      : each node slot v takes the argmin over eligible entries of column v of (weight, -age[u])
           -> key, parent; writes candidate flag, cost, and the output value
  C      : every slot finds the argmin-cost candidate; that slot gets marked (one Prim extraction per loop)
Node slots are the number tokens of `initial_trace: [0 1 ... n-1]`; the output pi is written there.
"""
import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "net")

DK = 4                    # every head uses dk = 4, so 1/sqrt(dk) = 1/2 exactly
QS = math.sqrt(DK)
KMAX = 40                 # largest row/column index the parser counts to
NOUT = 64                 # output alphabet: node numbers 0..NOUT-1
AGEM = 128                # tie-break multiplier: cost = AGEM*w - age  (needs R < AGEM)

# ---------------------------------------------------------------- dims
DIMS = []


def dim(name):
    DIMS.append(name)
    return len(DIMS) - 1


ONE, NUM, LB, RB, SP = dim("one"), dim("num"), dim("lb"), dim("rb"), dim("sp")
IV, IV2, W, POS = dim("iv"), dim("iv2"), dim("w"), dim("pos")
# P1
rF, rL, rB, rRB = dim("rF"), dim("rL"), dim("rB"), dim("rRB")
BD, FIRSTLB, STOK, ENTRY, SLOT = dim("b"), dim("firstlb"), dim("stok"), dim("entry"), dim("slot")
EMLB, EMRB, EMSP = dim("emlb"), dim("emrb"), dim("emsp")
# P2
rS, rMr, rMc = dim("rS"), dim("rMr"), dim("rMc")
U, V, V2, START, EDGE = dim("u"), dim("v"), dim("v2"), dim("start"), dim("edge")
# A
rAu, rAv = dim("rAu"), dim("rAv")
ELIG, AGEU = dim("elig"), dim("ageu")
# B
rPAR, rKEY, rVAL = dim("rPAR"), dim("rKEY"), dim("rVAL")
CAND, COST, OUTV = dim("cand"), dim("cost"), dim("out")
# C
rX, rANY = dim("rX"), dim("rANY")
MARK, AGE = dim("mark"), dim("age")
D = len(DIMS)

# ---------------------------------------------------------------- builders


class Layer:
    def __init__(self):
        self.heads = []      # (q comps, k comps, v list, mask)
        self.units = []      # (in dict, bias, out dict)
        self.b2 = {}

    def head(self, q, k, v, mask="none"):
        assert len(q) == len(k) <= DK
        self.heads.append((q, k, v, mask))

    def unit(self, inp, bias, out):
        self.units.append((inp, bias, out))

    # helpers ------------------------------------------------------
    def clear(self, d):
        """d -= relu(d)   (d is non-negative)"""
        self.unit({d: 1.0}, 0.0, {d: -1.0})

    def copy_set(self, src, dst):
        """dst := src (both non-negative); src is cleared."""
        self.unit({src: 1.0}, 0.0, {dst: 1.0, src: -1.0})
        self.clear(dst)

    def step(self, inp, bias, out, gain=1.0):
        """adds out * clip(gain*(inp.x + bias), 0, 1): 1 when inp.x+bias >= 1/gain, 0 when <= 0."""
        g = {k: v * gain for k, v in inp.items()}
        self.unit(g, bias * gain, dict(out))
        self.unit(g, bias * gain - 1.0, {k: -v for k, v in out.items()})

    def tri(self, inp, bias, out):
        """adds out * max(0, 1-|x|), x = inp.x + bias (equality test for integers)."""
        self.unit(inp, bias + 1.0, dict(out))
        self.unit(inp, bias, {k: -2 * v for k, v in out.items()})
        self.unit(inp, bias - 1.0, dict(out))


def add(dct, k, v):
    dct[k] = dct.get(k, 0.0) + v


LAYERS = []
BIG_POS = 1e5   # gate for position comparisons (pos <= 8191)

# ---------------------------------------------------------------- P1
P1 = Layer()
LAYERS.append(P1)
b, Lm = 64.0, 1e8
# first '[' in the prompt (global), last '[' (global), most recent '[' (causal), last ']' (global)
P1.head([{ONE: -b}, {ONE: Lm}], [{POS: 1}, {LB: 1}], [(POS, rF, 1.0)])
P1.head([{ONE: b}, {ONE: Lm}], [{POS: 1}, {LB: 1}], [(POS, rL, 1.0)])
P1.head([{ONE: b}, {ONE: Lm}], [{POS: 1}, {LB: 1}], [(POS, rB, 1.0)], mask="causal")
P1.head([{ONE: b}, {ONE: Lm}], [{POS: 1}, {RB: 1}], [(POS, rRB, 1.0)])
M = BIG_POS
# e1 = [num and pos > F]; e2 = [num and pos > L]; ENTRY = e1-e2; SLOT = e2; STOK = num - e1
P1.step({POS: 1, rF: -1, NUM: M}, -M, {ENTRY: 1, STOK: -1})
P1.step({POS: 1, rL: -1, NUM: M}, -M, {ENTRY: -1, SLOT: 1})
P1.unit({NUM: 1}, 0.0, {STOK: 1})
P1.step({rF: 1, POS: -1, LB: M}, 1 - M, {FIRSTLB: 1})          # lb and pos <= F
P1.step({POS: 1, rL: -1, LB: M}, 1 - M, {EMLB: 1})             # lb and pos >= L
P1.step({POS: 1, rL: -1, RB: M}, -M, {EMRB: 1})                # rb and pos > L
P1.step({POS: 1, rL: -1, SP: M}, -M, {EMSP: 1})                # sp and pos > L
P1.step({POS: 1, rRB: -1, SP: M}, -M, {EMSP: -1})              #   minus (sp and pos > last ']')
for d_ in (ENTRY, SLOT, STOK, FIRSTLB, EMLB, EMRB, EMSP):
    P1.clear(d_)
P1.copy_set(rB, BD)
for d_ in (rF, rL, rRB):
    P1.clear(d_)

# ---------------------------------------------------------------- P2
P2 = Layer()
LAYERS.append(P2)
Lm = 100.0
P2.head([{ONE: Lm}], [{STOK: 1}], [(IV, rS, 1.0)])                             # start node s
P2.head([{ONE: Lm}], [{LB: 1}], [(FIRSTLB, rMr, 1.0)], mask="causal")          # 1/(u+2)
# same row segment (largest b among num/lb tokens in the causal prefix): 1/(v+2)
P2.head([{ONE: 64.0}, {ONE: 1e7}], [{BD: 1}, {NUM: 1, LB: 1}], [(LB, rMc, 1.0)], mask="causal")
G = 2e4
for kk in range(1, KMAX + 1):
    t = 1.0 / (kk + 1.5)
    P2.step({rMr: -1}, t, {U: 1}, gain=G)
    P2.step({rMc: -1}, t, {V: 1, V2: 2 * kk - 1}, gain=G)
M2 = 1e4
P2.tri({IV: 1, rS: -1, SLOT: M2}, -M2, {START: 1})
P2.step({W: 1, ENTRY: 1e7}, -1e7, {EDGE: 1})                               # entry and w >= 1 (milli)
for d_ in (U, V, V2, START, EDGE, rS, rMr, rMc):
    P2.clear(d_)

# ---------------------------------------------------------------- A
A = Layer()
LAYERS.append(A)
bA, LA = 64.0, 1e10
# entry fetches age of node slot u (row) and v (column):  score = -bA (iv - u)^2 + LA slot
A.head([{U: 2 * bA}, {ONE: -bA}, {ONE: LA}], [{IV: 1}, {IV2: 1}, {SLOT: 1}], [(AGE, rAu, 1.0)])
A.head([{V: 2 * bA}, {ONE: -bA}, {ONE: LA}], [{IV: 1}, {IV2: 1}, {SLOT: 1}], [(AGE, rAv, 1.0)])
# ELIG = edge and age_u >= age_v + 1
A.step({rAu: 1, rAv: -1, EDGE: 1e4}, -1e4, {ELIG: 1})
A.copy_set(rAu, AGEU)
A.clear(ELIG)
A.clear(rAv)

# ---------------------------------------------------------------- B
B = Layer()
LAYERS.append(B)
bB, LB_, gB = 1e8, 2e7, 64.0
# slot j: argmin over eligible entries in column j of AGEM*w - age_u
B.head([{IV: 2 * bB}, {ONE: -bB}, {ONE: LB_}, {ONE: -gB * AGEM}],
       [{V: 1}, {V2: 1}, {ELIG: 1, AGEU: gB / LB_}, {W: 1}],
       [(U, rPAR, 1.0), (W, rKEY, 1.0), (ELIG, rVAL, 1.0)])
# CAND = slot and not marked and (valid or start)
B.step({rVAL: 1, START: 1, MARK: -1, SLOT: 2}, -2, {CAND: 1})
# COST = 64*key*valid + j
B.unit({rKEY: 64.0, rVAL: 1e8}, -1e8, {COST: 1})
B.unit({IV: 1}, 0.0, {COST: 1})
# OUT = slot ? (marked and not start ? parent : j) : 0
MB = 1e4
gP = {MARK: MB, SLOT: MB, START: -MB}
B.unit({**gP, rPAR: 1}, -2 * MB, {OUTV: 1})
B.unit({IV: 1, SLOT: MB}, -MB, {OUTV: 1})
B.unit({**gP, IV: 1}, -2 * MB, {OUTV: -1})
for d_ in (CAND, COST, OUTV, rPAR, rKEY, rVAL):
    B.clear(d_)

# ---------------------------------------------------------------- C
C = Layer()
LAYERS.append(C)
LC, gC = 1e8, 64.0
C.head([{ONE: LC}, {ONE: -gC}], [{CAND: 1}, {COST: 1}], [(IV, rX, 1.0), (CAND, rANY, 1.0)])
# SEL = slot and any and j == X ;  MARK += SEL ; AGE += MARK + SEL
MC = 1e3
C.tri({IV: 1, rX: -1, rANY: MC, SLOT: MC}, -2 * MC, {MARK: 1, AGE: 1})
C.unit({MARK: 1}, 0.0, {AGE: 1})
C.clear(rX)
C.clear(rANY)

# ---------------------------------------------------------------- assemble
ALPHABET = ["", "[", "]", " "] + [str(k) for k in range(NOUT)]


def build():
    E0 = np.zeros((257, D))
    E0[:, ONE] = 1
    E0[256, NUM] = 1
    E0[ord("["), LB] = 1
    E0[ord("]"), RB] = 1
    E0[ord(" "), SP] = 1
    E1 = np.zeros((1000, D))
    iv = np.arange(1000.0)
    E1[:, IV], E1[:, IV2], E1[:, W] = iv, iv * iv, 1000 * iv
    E2 = np.zeros((1000, D))
    E2[:, W] = np.arange(1000.0)
    E3 = np.zeros((8192, D))
    E3[:, POS] = np.arange(8192.0)
    Wt = {"emb.0": E0, "emb.1": E1, "emb.2": E2, "emb.3": E3}
    cfg_layers = []
    for li, L in enumerate(LAYERS):
        nh = len(L.heads)
        dvs = [len(h[2]) for h in L.heads]
        Wq, Wk = np.zeros((D, DK * nh)), np.zeros((D, DK * nh))
        Wv, Wo = np.zeros((D, sum(dvs))), np.zeros((sum(dvs), D))
        v0 = 0
        for hi, (q, k, v, mask) in enumerate(L.heads):
            for c, (qc, kc) in enumerate(zip(q, k)):
                for dd, x in qc.items():
                    Wq[dd, hi * DK + c] += x * QS
                for dd, x in kc.items():
                    Wk[dd, hi * DK + c] += x
            for c, (src, dst, x) in enumerate(v):
                Wv[src, v0 + c] = 1.0
                Wo[v0 + c, dst] = x
            v0 += len(v)
        nu = len(L.units)
        W1, b1, W2, b2 = np.zeros((D, nu)), np.zeros(nu), np.zeros((nu, D)), np.zeros(D)
        for ui, (inp, bias, out) in enumerate(L.units):
            for dd, x in inp.items():
                W1[dd, ui] += x
            b1[ui] = bias
            for dd, x in out.items():
                W2[ui, dd] += x
        for dd, x in L.b2.items():
            b2[dd] += x
        if nh:
            Wt.update({f"L{li}.Wq": Wq, f"L{li}.Wk": Wk, f"L{li}.Wv": Wv, f"L{li}.Wo": Wo})
        Wt.update({f"L{li}.W1": W1, f"L{li}.b1": b1, f"L{li}.W2": W2, f"L{li}.b2": b2})
        cfg_layers.append({"heads": [{"dk": DK, "dv": dv, "mask": h[3]} for h, dv in zip(L.heads, dvs)],
                           "d_ff": nu})
    # readout
    Go = 1e4
    Wout, bout = np.zeros((D, len(ALPHABET))), np.zeros(len(ALPHABET))
    for a, fl in ((1, EMLB), (2, EMRB), (3, EMSP)):
        Wout[fl, a] = Go
        bout[a] = -Go / 2
    for kk in range(NOUT):
        a = 4 + kk
        Wout[SLOT, a] = Go
        Wout[OUTV, a] = 2.0 * kk
        bout[a] = -Go / 2 - kk * kk
    Wt["out.W"], Wt["out.b"] = Wout, bout
    cfg = {"d_model": D, "fields": ["cls", "ival", "fval", "pos"], "layers": cfg_layers,
           "loops": {"const": 2, "lin": 1, "div": 24, "log2": 0}, "alphabet": ALPHABET}
    os.makedirs(OUT_DIR, exist_ok=True)
    json.dump(cfg, open(os.path.join(OUT_DIR, "network.json"), "w"), indent=1)
    sp = {}
    for k_, a_ in Wt.items():
        flat = a_.ravel()
        nz = np.nonzero(flat)[0]
        sp[k_ + ".shape"] = np.array(a_.shape, dtype=np.int64)
        sp[k_ + ".idx"] = nz.astype(np.int64)
        sp[k_ + ".val"] = flat[nz]
    np.savez(os.path.join(OUT_DIR, "weights.npz"), **sp)
    tot = sum(a_.size for a_ in Wt.values())
    nnz = sum(int(np.count_nonzero(a_)) for a_ in Wt.values())
    print(f"d_model={D} layers={len(LAYERS)} params total={tot} nonzero={nnz}")


if __name__ == "__main__":
    build()
