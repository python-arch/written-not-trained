"""Writes every weight of the lcs_length looped transformer. No data, no fitting.

Run: python3 -I work/write_weights.py   -> work/net/{network.json, weights.npz, tokenizer.py}
"""
import json
import math
import os
import shutil

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "net")

NCHAR = 256          # char field: min(ord(c), 255)
NPOS = 2048          # position field: min(i, NPOS-1)
TB = 24              # thresholds for '[' and ']' counts
TS = 160             # thresholds for space counts
K = 1e5              # step sharpness for counting
LAM = 100.0          # attention sharpness for exact-index lookups
BIG = 1e7            # attention bonus for the right kind of key token
M = 20.0             # gate size in the DP MLP (LCS values are <= 5)
SQ = 40              # squares are exact on integers 0..SQ
R_LOOPS = 12

LIT = ["[", "]", " ", ",", "-"] + [str(d) for d in range(10)]
ALPHABET = [""] + LIT

# ---------------------------------------------------------------- residual layout
names = []


def slot(name):
    names.append(name)
    return len(names) - 1


ONE = slot("one")
IS = {ch: slot("is_" + ch) for ch in LIT}
DIG = slot("is_digit")
DVAL = slot("dval")
POS = slot("pos")
POS2 = slot("pos2")
THR = [slot(f"thr{t}") for t in range(1, TS + 1)]      # THR[t-1] = (t-0.5)/(p+1)
# persistent computed slots
B, E, S, PM = slot("B"), slot("E"), slot("S"), slot("pm")
C_, BSQ, CSQ = slot("c"), slot("Bsq"), slot("csq")
IB2, IE1, ID2, IB4P, IB3P, ID1P, GR = [slot(x) for x in ("IB2", "IE1", "ID2", "IB4p", "IB3p", "ID1p", "gr")]
MATCH, VALID, ISA, INTR = slot("match"), slot("valid"), slot("isA"), slot("intr")
CVAL = slot("C")
BD = [slot(f"b{d}") for d in range(3)]
OLIT = {ch: slot("lit_" + ch) for ch in LIT}
# scratch slots (attention outputs; cleared by the same layer's MLP)
SC_OPEN, SC_CLOSE, SC_SPACE, SC_PM = slot("sc_open"), slot("sc_close"), slot("sc_space"), slot("sc_pm")
SC_SLO = slot("sc_Slo")
SC_KR, SC_KC = slot("sc_kr"), slot("sc_kc")
SC_CU, SC_CL, SC_CD = slot("sc_Cu"), slot("sc_Cl"), slot("sc_Cd")
D = len(names)

# ---------------------------------------------------------------- embeddings
emb_char = np.zeros((NCHAR, D))
for code in range(NCHAR):
    ch = chr(code)
    emb_char[code, ONE] = 1.0
    if ch in IS:
        emb_char[code, IS[ch]] = 1.0
    if ch.isdigit() and code < 128:
        emb_char[code, DIG] = 1.0
        emb_char[code, DVAL] = float(int(ch))
emb_pos = np.zeros((NPOS, D))
for p in range(NPOS):
    emb_pos[p, POS] = p
    emb_pos[p, POS2] = p * p
    for t in range(1, TS + 1):
        emb_pos[p, THR[t - 1]] = (t - 0.5) / (p + 1)


# ---------------------------------------------------------------- builders
class MLP:
    def __init__(self):
        self.units = []          # (input dict, bias, output dict)

    def unit(self, inp, bias, out):
        self.units.append((dict(inp), float(bias), dict(out)))

    def copy_to(self, src, dst, coef=1.0):
        """dst += coef * src (src any sign)."""
        self.unit({src: 1.0}, 0.0, {dst: coef})
        self.unit({src: -1.0}, 0.0, {dst: -coef})

    def clear(self, s):
        self.copy_to(s, s, -1.0)

    def step(self, inp, bias, dst, coef=1.0):
        """dst += coef * clamp(inp.h + bias, 0, 1)."""
        self.unit(inp, bias, {dst: coef})
        self.unit(inp, bias - 1.0, {dst: -coef})

    def tri(self, inp, bias, dst):
        """dst += max(0, 1 - |x|), x = inp.h + bias (exact on integers)."""
        self.unit(inp, bias + 1.0, {dst: 1.0})
        self.unit(inp, bias, {dst: -2.0})
        self.unit(inp, bias - 1.0, {dst: 1.0})

    def square(self, inp, bias, dst):
        """dst += x^2 for x = inp.h + bias in [0, SQ] (exact on integers)."""
        for k in range(SQ):
            self.unit(inp, bias - k, {dst: 1.0 if k == 0 else 2.0})

    def arrays(self):
        f = len(self.units)
        W1, b1, W2 = np.zeros((D, f)), np.zeros(f), np.zeros((f, D))
        for j, (inp, bias, out) in enumerate(self.units):
            for i, v in inp.items():
                W1[i, j] += v
            b1[j] = bias
            for i, v in out.items():
                W2[j, i] += v
        return W1, b1, W2, np.zeros(D)


def head(q, k, v_src, v_dst, mask="none"):
    """q, k: lists of dicts (one per score dim), score = sum_i q_i.h * k_i.h. One value dim."""
    return {"q": q, "k": k, "v": [(v_src, v_dst)], "mask": mask}


def head_arrays(heads):
    dks = [len(h["q"]) for h in heads]
    dvs = [len(h["v"]) for h in heads]
    Wq, Wk = np.zeros((D, sum(dks))), np.zeros((D, sum(dks)))
    Wv, Wo = np.zeros((D, sum(dvs))), np.zeros((sum(dvs), D))
    a = b = 0
    for h, dk, dv in zip(heads, dks, dvs):
        sc = math.sqrt(dk)                    # undo the runner's 1/sqrt(dk)
        for i in range(dk):
            for s, val in h["q"][i].items():
                Wq[s, a + i] += val * sc
            for s, val in h["k"][i].items():
                Wk[s, a + i] += val
        for i, (src, dst) in enumerate(h["v"]):
            Wv[src, b + i] = 1.0
            Wo[b + i, dst] = 1.0
        a += dk
        b += dv
    cfg = [{"dk": dk, "dv": dv, "mask": h["mask"]} for h, dk, dv in zip(heads, dks, dvs)]
    return cfg, (Wq, Wk, Wv, Wo)


def overwrite(mlp, dst):
    """Start an overwrite of dst: subtract its current value (targets are added by other units)."""
    mlp.clear(dst)


layers = []

# ---------------------------------------------------------------- L0: counts and previous char
h_count = {"q": [{}], "k": [{}], "v": [(IS["["], SC_OPEN), (IS["]"], SC_CLOSE), (IS[" "], SC_SPACE)],
           "mask": "causal"}                                  # uniform over the prefix: means
h_prev = head([{POS: 1.0, ONE: -1.0}, {ONE: 1.0}], [{POS: 2 * LAM}, {POS2: -LAM}], IS["-"], SC_PM)
m = MLP()
for dst in (B, E, S, PM):
    overwrite(m, dst)
for t in range(1, TB + 1):
    m.step({SC_OPEN: K, THR[t - 1]: -K}, 0.0, B)
    m.step({SC_CLOSE: K, THR[t - 1]: -K}, 0.0, E)
for t in range(1, TS + 1):
    m.step({SC_SPACE: K, THR[t - 1]: -K}, 0.0, S)
m.step({SC_PM: K}, -0.5 * K, PM)
for s in (SC_OPEN, SC_CLOSE, SC_SPACE, SC_PM):
    m.clear(s)
layers.append(([h_count, h_prev], m))

# ---------------------------------------------------------------- L1: column index and indicators
h_lo = head([{ONE: 1.0}], [{IS["["]: BIG, POS: LAM}], S, SC_SLO, mask="causal")   # latest '[' : its space count
m = MLP()
for dst in (C_, BSQ, CSQ, IB2, IE1, ID2, IB4P, IB3P, ID1P, GR):
    overwrite(m, dst)
m.copy_to(S, C_)
m.copy_to(SC_SLO, C_, -1.0)
m.square({B: 1.0}, 0.0, BSQ)
m.square({S: 1.0, SC_SLO: -1.0}, 0.0, CSQ)
m.tri({B: 1.0}, -2.0, IB2)
m.tri({E: 1.0}, -1.0, IE1)
m.tri({B: 1.0, E: -1.0}, -2.0, ID2)
m.step({B: 1.0}, -3.0, IB4P)
m.step({B: 1.0}, -2.0, IB3P)
m.step({B: 1.0, E: -1.0}, 0.0, ID1P)
m.tri({B: 1.0}, -4.0, GR)
m.clear(SC_SLO)
layers.append(([h_lo], m))

# ---------------------------------------------------------------- L2: fetch key[r], key[c]; flags
keysel = {IB2: BIG, IE1: BIG, DIG: BIG, CSQ: -LAM}
h_kr = head([{B: 1.0, ONE: -4.0}, {ONE: 1.0}], [{C_: 2 * LAM}, keysel], DVAL, SC_KR)
h_kc = head([{C_: 1.0}, {ONE: 1.0}], [{C_: 2 * LAM}, keysel], DVAL, SC_KC)
m = MLP()
for dst in (MATCH, VALID, ISA, INTR):
    overwrite(m, dst)
m.tri({SC_KR: 1.0, SC_KC: -1.0}, 0.0, MATCH)
m.unit({ID2: 1.0, IB4P: 1.0, DIG: 1.0, PM: -1.0}, -2.0, {VALID: 1.0})
m.unit({ID2: 1.0, IB4P: 1.0, DIG: 1.0}, -2.0, {ISA: 1.0})
m.step({IB3P: 2.0, ID1P: 1.0, IS["]"]: 1.0}, -2.0, INTR)
m.clear(SC_KR)
m.clear(SC_KC)
layers.append(([h_kr, h_kc], m))

# ---------------------------------------------------------------- L3: one DP step, b, output letters
anc = {CSQ: -LAM, BSQ: -LAM, ISA: BIG}


def dp_head(dr, dc, dst):
    return head([{B: 1.0, ONE: -dr}, {C_: 1.0, ONE: -dc}, {ONE: 1.0}],
                [{B: 2 * LAM}, {C_: 2 * LAM}, anc], CVAL, dst)


heads = [dp_head(1, 0, SC_CU), dp_head(0, 1, SC_CL), dp_head(1, 1, SC_CD)]
m = MLP()
for dst in [CVAL] + BD + list(OLIT.values()):
    overwrite(m, dst)
Cu, Cl, Cd, v, mt, gr = SC_CU, SC_CL, SC_CD, VALID, MATCH, GR
# C = v*(m*(Cd'+1) + (1-m)*max(Cu', Cl)), primes: 0 on row 0
m.unit({mt: M, v: M}, 1.0 - 2 * M, {CVAL: 1.0})
m.unit({Cd: 1.0, mt: M, v: M, gr: -M}, -2 * M, {CVAL: 1.0})
m.unit({Cl: 1.0, v: M, mt: -M}, -M, {CVAL: 1.0})
m.unit({Cu: 1.0, Cl: -1.0, v: M, mt: -M, gr: -M}, -M, {CVAL: 1.0})
# b0 = v*m ; b2 = v*(1-m)*[Cl > Cu'] ; b1 = v*(1-m) - b2
m.unit({v: 1.0, mt: 1.0}, -1.0, {BD[0]: 1.0})
m.unit({v: 1.0, mt: -1.0}, 0.0, {BD[1]: 1.0})
for inp, bias in (({Cl: 1.0, Cu: -1.0, v: M, mt: -M, gr: -M}, -M),          # rows below the first
                  ({Cl: 1.0, v: M, mt: -M, gr: M}, -2 * M)):                # first row: Cu' = 0
    m.unit(inp, bias, {BD[2]: 1.0, BD[1]: -1.0})
    m.unit(inp, bias - 1.0, {BD[2]: -1.0, BD[1]: 1.0})
# letters copied from the prompt's trace (all but the cells being answered)
for ch in LIT:
    m.unit({INTR: 1.0, IS[ch]: 1.0, v: -1.0}, -1.0, {OLIT[ch]: 1.0})
for s in (SC_CU, SC_CL, SC_CD):
    m.clear(s)
layers.append((heads, m))

# ---------------------------------------------------------------- readout
Wout, bout = np.zeros((D, len(ALPHABET))), np.zeros(len(ALPHABET))
bout[0] = 0.5
for ch in LIT:
    Wout[OLIT[ch], ALPHABET.index(ch)] = 1.0
for d in range(3):
    Wout[BD[d], ALPHABET.index(str(d))] = 1.0

# ---------------------------------------------------------------- write
os.makedirs(OUT, exist_ok=True)
W = {"emb.0": emb_char, "emb.1": emb_pos, "out.W": Wout, "out.b": bout}
cfg_layers = []
for l, (hs, mlp) in enumerate(layers):
    W1, b1, W2, b2 = mlp.arrays()
    W.update({f"L{l}.W1": W1, f"L{l}.b1": b1, f"L{l}.W2": W2, f"L{l}.b2": b2})
    hcfg = []
    if hs:
        hcfg, (Wq, Wk, Wv, Wo) = head_arrays(hs)
        W.update({f"L{l}.Wq": Wq, f"L{l}.Wk": Wk, f"L{l}.Wv": Wv, f"L{l}.Wo": Wo})
    cfg_layers.append({"heads": hcfg, "d_ff": W1.shape[1]})
cfg = {"d_model": D, "fields": ["char", "position"], "layers": cfg_layers,
       "loops": {"const": R_LOOPS, "lin": 0, "div": 1, "log2": 0}, "alphabet": ALPHABET}
json.dump(cfg, open(os.path.join(OUT, "network.json"), "w"), indent=1)
np.savez(os.path.join(OUT, "weights.npz"), **W)
shutil.copy(os.path.join(HERE, "tokenizer.py"), os.path.join(OUT, "tokenizer.py"))
tot = sum(a.size for a in W.values())
nz = sum(int(np.count_nonzero(a)) for a in W.values())
print(f"d_model={D} layers={len(layers)} d_ff={[c['d_ff'] for c in cfg_layers]} params={tot} nonzero={nz}")
