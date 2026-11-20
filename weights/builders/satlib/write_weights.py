"""Writes every weight of the looped-transformer 3-SAT decider. Nothing is trained or fitted: every number below is
set by hand-derived rules.

Run:  python3 -I work/write_weights.py   ->  work/net/network.json, work/net/weights.npz  (work/net/tokenizer.py is
hand written) and work/dims.json (names of the residual dimensions, for my own debugging).

What the network computes: DPLL with chronological backtracking, parallel unit propagation and Jeroslow-Wang
branching; one loop of the block = one DPLL step.
  * Token positions 1..VMAX (inside the instruction line) are "cells". Cell v holds variable v (T, F, LEV, DEC) and
    decision level v (FLG = its decision has been flipped, DV = its decision variable). The global scalars curL
    (current level), SATD and UNSD (halted with SAT / UNSAT) are held identically at every position.
  * One step:  L1 each literal token reads its variable's cell -> literal true / false.
               L2 a same-line mean evaluates every clause line -> per literal token: eligible (open literal of an
                  open clause), forced (open literal of a unit clause); per clause: conflict, active.
               L3 per cell: forced-positive / forced-negative flags and the JW score S/(S+4) of its variable;
                  global priority OR: conflict? else forced? else open clause? else (SAT) the header's V.
               L4 one selector head: if conflict, the highest unflipped level (and its decision variable); else the
                  unassigned cell with the largest JW score (ties: lowest index) and its preferred sign.
                  Mode = backtrack | unsat | propagate | sat | decide (or nothing once halted).
               L5 the state update.
  * A variable forced both ways is set T = F = 1, which makes both its literals false, so the next step sees a
    conflict clause.
  * Readout: position 0 writes "SAT" / "UNSAT" / "UNKNOWN"; when SAT, cell k (k <= V) writes " k" or " -k".
"""
import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "net")
VMAX = 50
NF_LINE = 10                 # frequencies of the line-index code (lines < 1024)
T_KIND, T_NUM, T_POS, T_LINE, T_COL = 3 * 257, 2001, 4096, 1024, 64
LN2 = math.log(2.0)


class Dims:
    def __init__(self):
        self.idx = {}

    def add(self, name, k=1):
        for i in range(k):
            self.idx[name if k == 1 else f"{name}{i}"] = len(self.idx)

    def __getitem__(self, name):
        return self.idx[name]

    def __len__(self):
        return len(self.idx)


D = Dims()
STATIC = ["ONE", "ISNUM", "ZERO", "ISLIT", "SIGN", "NUMVAL", "POS0", "ISCELL", "CELLIDX", "COL", "INFS", "LINE1"]
STATE = ["T", "F", "LEV", "DEC", "FLG", "DV", "CURL", "SATD", "UNSD"]
TRANS = ["TL", "FL", "LT", "LF", "URAW", "MLT", "MLF", "MZ", "ELIG", "ELIGP", "FORCED", "FPT", "FNT", "CONFT", "ACTT",
         "SC", "SCP", "RAWP", "RAWN", "CC", "AF", "AA", "VV", "FPS", "FNS", "POSPREF", "INR",
         "HASU", "SEL", "W", "CHS", "GE", "ISW", "ISSEL", "GTB", "ISNEXT", "MB", "MU", "MP", "MS", "MD",
         "CCS", "AFS", "AAS", "CHSS"]
for nm in STATIC + STATE + TRANS:
    D.add(nm)
D.add("VALOH", VMAX + 1)       # one-hot of a literal's variable (index 0 unused)
D.add("POSOH", VMAX + 1)       # one-hot of a cell's index (index 0 unused)
D.add("PEL", 2 * NF_LINE)
d = len(D)

# ----------------------------------------------------------------------------------------------- embedding
E0 = np.zeros((T_KIND, d))     # kind = leading_ws * 257 + base
E0[:, D["ONE"]] = 1
for ws in range(3):
    E0[ws * 257 + 256, D["ISNUM"]] = 1

E1 = np.zeros((T_NUM, d))      # number = 1 + 2 * value + sign
for x in range(1000):
    for s in range(2):
        r = 1 + 2 * x + s
        E1[r, D["NUMVAL"]] = x
        E1[r, D["SIGN"]] = s
        if x == 0:
            E1[r, D["ZERO"]] = 1
        if 1 <= x <= VMAX:
            E1[r, D["ISLIT"]] = 1
            E1[r, D[f"VALOH{x}"]] = 1

E2 = np.zeros((T_POS, d))      # position
E2[0, D["POS0"]] = 1
for p in range(1, VMAX + 1):
    E2[p, D["ISCELL"]] = 1
    E2[p, D["CELLIDX"]] = p
    E2[p, D[f"POSOH{p}"]] = 1

E3 = np.zeros((T_LINE, d))     # line index
for L in range(T_LINE):
    for f in range(NF_LINE):
        E3[L, D[f"PEL{2 * f}"]] = math.cos(math.pi * L / 2 ** f)
        E3[L, D[f"PEL{2 * f + 1}"]] = math.sin(math.pi * L / 2 ** f)
    E3[L, D["INFS"]] = 1.0 if L >= 1 else 0.0       # not in the first (instruction) line
    E3[L, D["LINE1"]] = 1.0 if L == 1 else 0.0

E4 = np.zeros((T_COL, d))      # column (token index within its line)
E4[:, D["COL"]] = np.arange(T_COL)


# ----------------------------------------------------------------------------------------------- builders
class Head:
    def __init__(self, dk, dv, mask="none"):
        self.dk, self.dv, self.mask = dk, dv, mask
        self.Wq, self.Wk = np.zeros((d, dk)), np.zeros((d, dk))
        self.Wv, self.Wo = np.zeros((d, dv)), np.zeros((dv, d))

    def q(self, name, j, w):     # the score is exactly q.k: the runner's 1/sqrt(dk) is undone here
        self.Wq[D[name], j] += w * math.sqrt(self.dk)

    def k(self, name, j, w):
        self.Wk[D[name], j] += w

    def v(self, name, j, w=1.0):
        self.Wv[D[name], j] += w

    def o(self, j, name, w=1.0):
        self.Wo[j, D[name]] += w


class MLP:
    def __init__(self):
        self.units = []          # (inputs {dim: w}, bias, outputs {dim: w}); hidden = relu(inputs . h + bias)

    def u(self, inp, b, out):
        self.units.append((dict(inp), float(b), dict(out)))

    def hat(self, inp, b, out):  # out += [x == 0] for an integer x = inp.h + b
        self.u(inp, b + 1, {out: 1}); self.u(inp, b, {out: -2}); self.u(inp, b - 1, {out: 1})

    def ge0(self, inp, b, out):  # out += [x >= 0] for an integer x
        self.u(inp, b + 1, {out: 1}); self.u(inp, b, {out: -1})

    def snap(self, inp, b, out):  # out += clip(x, 0, 1)
        self.u(inp, b, {out: 1}); self.u(inp, b - 1, {out: -1})

    # margin-robust versions for inputs that are integers up to small float error
    def sbin(self, inp, b, out, w=1.0):  # out += w*[x >= 1/2] for x = inp.h + b near an integer <= 0 or >= 1
        i2 = {k_: 2 * v_ for k_, v_ in inp.items()}
        self.u(i2, 2 * b - 0.5, {out: w}); self.u(i2, 2 * b - 1.5, {out: -w})

    def rge0(self, inp, b, out):  # out += [x >= 0] for x near an integer
        i2 = {k_: 2 * v_ for k_, v_ in inp.items()}
        self.u(i2, 2 * b + 1.5, {out: 1}); self.u(i2, 2 * b + 0.5, {out: -1})

    def rhat(self, inp, b, out):  # out += [x == 0] for x near an integer
        i2 = {k_: 2 * v_ for k_, v_ in inp.items()}
        self.u(i2, 2 * b + 1.5, {out: 1}); self.u(i2, 2 * b + 0.5, {out: -1})
        self.u(i2, 2 * b - 0.5, {out: -1}); self.u(i2, 2 * b - 1.5, {out: 1})

    def clear(self, name, signed=True):  # out -= its current value
        self.u({name: 1}, 0, {name: -1})
        if signed:
            self.u({name: -1}, 0, {name: 1})


def plus(*ds):
    r = {}
    for x in ds:
        for k_, v_ in x.items():
            r[k_] = r.get(k_, 0) + v_
    return r


layers = []

# ---------------- L1: each literal token reads T, F of its variable's cell
h_look = Head(VMAX, 2)
for v in range(1, VMAX + 1):
    h_look.q(f"VALOH{v}", v - 1, 60.0); h_look.k(f"POSOH{v}", v - 1, 1.0)
h_look.v("T", 0); h_look.v("F", 1); h_look.o(0, "TL"); h_look.o(1, "FL")
m = MLP()
g = {"ISLIT": 1, "INFS": 1}                        # literal in the formula part
m.sbin(plus(g, {"TL": 1, "FL": -1, "SIGN": -1}), -2, "LT")      # positive literal: T and not F
m.sbin(plus(g, {"FL": 1, "TL": -1, "SIGN": 1}), -3, "LT")       # negative literal: F and not T
m.sbin(plus(g, {"FL": 1, "SIGN": -1}), -2, "LF")                # positive literal: F
m.sbin(plus(g, {"TL": 1, "SIGN": 1}), -3, "LF")                 # negative literal: T
m.clear("LT"); m.clear("LF")
m.rge0({"CURL": 1, "CELLIDX": -1, "ISCELL": 100}, -100, "URAW"); m.clear("URAW")   # cell index <= curL
for nm in ["T", "F", "DEC", "FLG", "SATD", "UNSD"]:             # re-snap the binary state
    m.sbin({nm: 1}, 0, nm); m.clear(nm)
m.clear("TL"); m.clear("FL")
layers.append(([h_look], m))

# ---------------- L2: same-line mean = clause evaluation; literal / clause flags
h_line = Head(2 * NF_LINE, 3)
for j in range(2 * NF_LINE):
    h_line.q(f"PEL{j}", j, 30.0); h_line.k(f"PEL{j}", j, 1.0)
for j, (a, b) in enumerate([("LT", "MLT"), ("LF", "MLF"), ("ZERO", "MZ")]):
    h_line.v(a, j); h_line.o(j, b)
m = MLP()
# with 4 tokens per clause line: 4*MLT = #true literals, 4*MLF = #false literals, 4*MZ = 1 on a clause line
el = {"INFS": 1, "MZ": 4, "ISLIT": 1, "MLT": -4, "LT": -1, "LF": -1}
m.sbin(el, -2, "ELIG"); m.clear("ELIG")
m.sbin(plus(el, {"SIGN": -1}), -2, "ELIGP"); m.clear("ELIGP")
fo = {"MLF": 4, "INFS": 3, "MZ": 12, "ISLIT": 3, "MLT": -12, "LT": -3, "LF": -3}
m.sbin(fo, -10, "FORCED"); m.clear("FORCED")
m.sbin(plus(fo, {"SIGN": -1}), -10, "FPT"); m.clear("FPT")
m.sbin(plus(fo, {"SIGN": 1}), -11, "FNT"); m.clear("FNT")
m.sbin({"MLF": 4, "INFS": 3, "MZ": 12, "ZERO": 3, "MLT": -12}, -11, "CONFT"); m.clear("CONFT")
m.sbin({"INFS": 1, "MZ": 4, "ZERO": 1, "MLT": -4}, -2, "ACTT"); m.clear("ACTT")
layers.append(([h_line], m))

# ---------------- L3: per-cell aggregation; global priority OR
A, FB = 100.0, 35.0
C0 = A + 30.0 + 5 * LN2         # sink at position 0: weight 4 x that of an open literal in a clause with one false
h_cell = Head(VMAX + 1, 4)
for v in range(1, VMAX + 1):
    h_cell.q(f"POSOH{v}", v - 1, A); h_cell.k(f"VALOH{v}", v - 1, 1.0)
h_cell.q("ONE", VMAX, 1.0)
h_cell.k("FORCED", VMAX, FB); h_cell.k("ELIG", VMAX, 30.0); h_cell.k("MLF", VMAX, 4 * LN2)
h_cell.k("ONE", VMAX, -3 * LN2); h_cell.k("POS0", VMAX, C0)
for j, (a, b) in enumerate([("ELIG", "SC"), ("ELIGP", "SCP"), ("FPT", "RAWP"), ("FNT", "RAWN")]):
    h_cell.v(a, j); h_cell.o(j, b)
h_pri = Head(1, 4)
h_pri.q("ONE", 0, 1.0)
for nm, w in [("CONFT", 1000.0), ("FORCED", 500.0), ("ACTT", 250.0), ("LINE1", 200.0), ("ISNUM", 200.0),
              ("COL", -40.0)]:
    h_pri.k(nm, 0, w)
for j, (a, b) in enumerate([("CONFT", "CC"), ("FORCED", "AF"), ("ACTT", "AA"), ("NUMVAL", "VV")]):
    h_pri.v(a, j); h_pri.o(j, b)
m = MLP()
m.snap({"RAWP": 100, "ISCELL": 200}, -200.5, "FPS"); m.clear("FPS")
m.snap({"RAWN": 100, "ISCELL": 200}, -200.5, "FNS"); m.clear("FNS")
m.snap({"SCP": 2000, "SC": -1000}, 2, "POSPREF"); m.clear("POSPREF")       # S+ >= S-
m.rge0({"VV": 1, "CELLIDX": -1, "ISCELL": 2000}, -2000, "INR"); m.clear("INR")  # cell index <= V
for a, b in [("CC", "CCS"), ("AF", "AFS"), ("AA", "AAS")]:
    m.sbin({a: 1}, 0, b); m.clear(b)
for nm in ["MLT", "MLF", "MZ", "SCP", "RAWP", "RAWN", "VV", "CC", "AF", "AA"]:
    m.clear(nm)
layers.append(([h_cell, h_pri], m))

# ---------------- L4: selector (conflict: highest unflipped level; else: JW argmax); compares and modes
MJ, PJ, TB = 8.0e6, 1.0e7, 200.0
h_sel = Head(2, 4)
h_sel.q("CCS", 0, 1.0)
h_sel.k("URAW", 0, 2500.0); h_sel.k("FLG", 0, -2500.0); h_sel.k("CELLIDX", 0, 40.0)
h_sel.q("ONE", 1, 1.0); h_sel.q("CCS", 1, -1.0)
h_sel.k("SC", 1, MJ); h_sel.k("ISCELL", 1, PJ + 51 * TB); h_sel.k("CELLIDX", 1, -TB)
h_sel.k("T", 1, -PJ); h_sel.k("F", 1, -PJ)
h_sel.v("URAW", 0); h_sel.v("FLG", 0, -1.0); h_sel.v("CELLIDX", 1); h_sel.v("DV", 2); h_sel.v("POSPREF", 3)
for j, nm in enumerate(["HASU", "SEL", "W", "CHS"]):
    h_sel.o(j, nm)
m = MLP()
m.rge0({"LEV": 1, "SEL": -1}, 0, "GE"); m.clear("GE")
m.rhat({"CELLIDX": 1, "W": -1}, 0, "ISW"); m.clear("ISW")
m.rhat({"CELLIDX": 1, "SEL": -1}, 0, "ISSEL"); m.clear("ISSEL")
m.rge0({"CELLIDX": 1, "SEL": -1}, -1, "GTB"); m.clear("GTB")
m.rhat({"CELLIDX": 1, "CURL": -1}, -1, "ISNEXT"); m.clear("ISNEXT")
m.sbin({"CHS": 1}, 0, "CHSS"); m.clear("CHSS")
act = {"SATD": -1, "UNSD": -1}                     # act = 1 - SATD - UNSD; its 1 is in each bias below
m.sbin(plus(act, {"CCS": 1, "HASU": 1}), -1, "MB")
m.sbin(plus(act, {"CCS": 1, "HASU": -1}), 0, "MU")
m.sbin(plus(act, {"CCS": -1, "AFS": 1}), 0, "MP")
m.sbin(plus(act, {"CCS": -1, "AFS": -1, "AAS": -1}), 1, "MS")
m.sbin(plus(act, {"CCS": -1, "AFS": -1, "AAS": 1}), 0, "MD")
for nm in ["MB", "MU", "MP", "MS", "MD", "SC", "CCS", "AFS", "AAS"]:
    m.clear(nm)
layers.append(([h_sel], m))

# ---------------- L5: state update
M = 100.0
m = MLP()
m.u({"FPS": 1, "MP": 1}, -1, {"T": 1})
m.u({"ISSEL": 1, "CHSS": 1, "MD": 1}, -2, {"T": 1})
m.u({"T": 1, "GE": 1, "MB": 1}, -2, {"T": -1})
m.u({"ISW": 1, "F": 1, "MB": 1}, -2, {"T": 1})
m.u({"FNS": 1, "MP": 1}, -1, {"F": 1})
m.u({"ISSEL": 1, "CHSS": -1, "MD": 1}, -1, {"F": 1})
m.u({"F": 1, "GE": 1, "MB": 1}, -2, {"F": -1})
m.u({"ISW": 1, "T": 1, "MB": 1}, -2, {"F": 1})
m.u({"CURL": 1, "MP": M, "FPS": M}, -2 * M, {"LEV": 1})
m.u({"CURL": 1, "MP": M, "FNS": M}, -2 * M, {"LEV": 1})
m.u({"CURL": 1, "MP": M, "FPS": M, "FNS": M}, -3 * M, {"LEV": -1})
m.u({"CURL": 1, "MD": M, "ISSEL": M}, 1 - 2 * M, {"LEV": 1})
m.u({"LEV": 1, "GE": M, "MB": M}, -2 * M, {"LEV": -1})
m.u({"SEL": 1, "ISW": M, "MB": M}, -2 * M, {"LEV": 1})
m.u({"ISSEL": 1, "MD": 1}, -1, {"DEC": 1})
m.u({"DEC": 1, "GE": 1, "MB": 1}, -2, {"DEC": -1})
m.u({"ISW": 1, "MB": 1}, -1, {"DEC": 1})
m.u({"ISSEL": 1, "MB": 1}, -1, {"FLG": 1})
m.u({"FLG": 1, "GTB": 1, "MB": 1}, -2, {"FLG": -1})
m.u({"SEL": 1, "DV": -1, "MD": M, "ISNEXT": M}, -2 * M, {"DV": 1})
m.u({"DV": 1, "SEL": -1, "MD": M, "ISNEXT": M}, -2 * M, {"DV": -1})
m.u({"MD": 1}, 0, {"CURL": 1})
m.u({"CURL": 1, "SEL": -1, "MB": M}, -M, {"CURL": -1})
m.u({"MS": 1}, 0, {"SATD": 1})
m.u({"MU": 1}, 0, {"UNSD": 1})
for nm in ["HASU", "SEL", "W", "CHS", "CHSS"]:
    m.clear(nm)
layers.append(([], m))

# ----------------------------------------------------------------------------------------------- readout
alphabet = ["", "SAT", "UNSAT", "UNKNOWN"]
for k in range(1, VMAX + 1):
    alphabet += [f" {k}", f" -{k}"]
Wout, bout = np.zeros((d, len(alphabet))), np.zeros(len(alphabet))
bout[0] = 0.5
Wout[D["POS0"], 1] = 2; Wout[D["SATD"], 1] = 2; bout[1] = -3
Wout[D["POS0"], 2] = 2; Wout[D["UNSD"], 2] = 2; bout[2] = -3
Wout[D["POS0"], 3] = 2; Wout[D["SATD"], 3] = -2; Wout[D["UNSD"], 3] = -2; bout[3] = -1
for k in range(1, VMAX + 1):
    for j, (fw, bb) in enumerate(((-2, -5), (2, -7))):
        c = 4 + 2 * (k - 1) + j
        Wout[D[f"POSOH{k}"], c] = 2; Wout[D["SATD"], c] = 2; Wout[D["INR"], c] = 2; Wout[D["F"], c] = fw
        bout[c] = bb


# ----------------------------------------------------------------------------------------------- assemble + save
def build():
    W = {"emb.0": E0, "emb.1": E1, "emb.2": E2, "emb.3": E3, "emb.4": E4, "out.W": Wout, "out.b": bout}
    cfg_layers = []
    for l, (heads, mlp) in enumerate(layers):
        if heads:
            W[f"L{l}.Wq"] = np.hstack([h.Wq for h in heads]); W[f"L{l}.Wk"] = np.hstack([h.Wk for h in heads])
            W[f"L{l}.Wv"] = np.hstack([h.Wv for h in heads]); W[f"L{l}.Wo"] = np.vstack([h.Wo for h in heads])
        nu = len(mlp.units)
        W1, b1, W2 = np.zeros((d, nu)), np.zeros(nu), np.zeros((nu, d))
        for i, (inp, b, out) in enumerate(mlp.units):
            for nm, w in inp.items():
                W1[D[nm], i] += w
            b1[i] = b
            for nm, w in out.items():
                W2[i, D[nm]] += w
        W[f"L{l}.W1"], W[f"L{l}.b1"], W[f"L{l}.W2"], W[f"L{l}.b2"] = W1, b1, W2, np.zeros(d)
        cfg_layers.append({"heads": [{"dk": h.dk, "dv": h.dv, "mask": h.mask} for h in heads], "d_ff": nu})
    cfg = {"d_model": d, "fields": ["kind", "number", "position", "line", "column"], "layers": cfg_layers,
           "loops": {"const": 0, "lin": 1, "div": 1, "log2": 0}, "alphabet": alphabet}
    return cfg, W


def main():
    cfg, W = build()
    os.makedirs(OUT, exist_ok=True)
    json.dump(cfg, open(os.path.join(OUT, "network.json"), "w"), indent=1)
    sparse = {}
    for k_, a in W.items():      # sparse storage: flat row-major indices and values
        a = np.asarray(a, dtype=np.float64)
        nz = np.flatnonzero(a)
        sparse[k_ + ".shape"] = np.array(a.shape, dtype=np.int64)
        sparse[k_ + ".idx"] = nz.astype(np.int64)
        sparse[k_ + ".val"] = a.ravel()[nz]
    np.savez_compressed(os.path.join(OUT, "weights.npz"), **sparse)
    json.dump(D.idx, open(os.path.join(HERE, "dims.json"), "w"))
    tot = sum(a.size for a in W.values()); nnz = sum(int(np.count_nonzero(a)) for a in W.values())
    print(f"d_model {d}, layers {len(layers)}, heads {[len(h) for h, _ in layers]}, "
          f"d_ff {[c['d_ff'] for c in cfg['layers']]}, params {tot}, nonzero {nnz}")


if __name__ == "__main__":
    main()
