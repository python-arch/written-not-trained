"""Writes every weight of the looped Sudoku transformer (no training, no data).

Run:  python3 -I work/write_weights.py      -> work/net/{network.json, weights.npz, tokenizer.py}

One loop of the block = one synchronous "tick" of a depth-first Sudoku solver, all 81 cells in parallel:

  state per cell: cand[9] (candidate bits), S[k][9] (stack snapshots, k < K), D one-hot (depth, same everywhere)
  L0  (MLP)        solved_v, U (unsolved), E (empty), g_v (smallest candidate)
  L1  (3 heads)    per unit (row/col/box) candidate counts and solved counts
      (MLP)        elim_v (a peer is solved as v), Hsum_v (hidden-single count), contra (empty cell / unit missing digit)
  L2  (MLP)        newc_v = cand_v and not elim_v and no other hidden single in this cell
  L3  (2 heads)    global counts (contradictions, changes, unsolved) and the selected guess cell (max priority)
      (MLP)        P (pop), Q (propagate), G (guess) flags; is_sel
  L4  (MLP)        cand, S, D updates:
                     P: cand <- S[D-1], D -= 1
                     Q: cand <- newc
                     G: S[D] <- cand minus guess at selected cell; selected cell cand <- {g}; D += 1
                     otherwise (solved): nothing

Every binary quantity is produced by a plateau step pstep(x) = relu(2x-.5) - relu(2x-1.5), which is exactly 0 for
x <= .25 and exactly 1 for x >= .75, so float noise never accumulates across loops.  Values that are written to the
residual stream are written with "set" semantics (new - old, old read through relu(x) - relu(-x)).
"""
import json
import os
import shutil

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "net")

K = 24                 # stack depth (DEV max observed: 13)
R_LOOPS = 40000        # ticks (DEV max observed: 5,585)
NPOS = 1024            # position table size (tokenizer clamps)
BETA = 40.0            # unit / global attention sharpness
BETA_SEL = 50.0        # selection attention sharpness (priority gaps are >= 1)
BIG = 1.0e4
PREFIX = len("Solve this Sudoku:\n")   # 19; used only by the weights (position embedding)


# ----------------------------------------------------------------------------------------------- residual layout
class Dims:
    def __init__(self):
        self.n = 0
        self.idx = {}

    def add(self, name, size=1):
        if size == 1:
            self.idx[name] = self.n
            self.n += 1
        else:
            for i in range(size):
                self.idx[(name, i)] = self.n
                self.n += 1

    def __getitem__(self, k):
        return self.idx[k]


dm = Dims()
for nm in ["one", "is_cell", "idx"]:
    dm.add(nm)
for nm in ["row", "col", "box"]:
    dm.add(nm, 9)
dm.add("cand", 9)
dm.add("solved", 9)
dm.add("U")
dm.add("E")
dm.add("g", 9)
for u in range(3):
    dm.add(("C", u), 9)          # keys (("C",u), v)
dm.add("Ssum", 9)
dm.add("elim", 9)
dm.add("Hsum", 9)
dm.add("contra")
dm.add("newc", 9)
for nm in ["c_con", "c_chg", "c_u", "p_sel", "P", "Q", "G", "is_sel"]:
    dm.add(nm)
for k in range(K):
    dm.add(("S", k), 9)          # keys (("S",k), v)
dm.add("D", K + 1)
D_MODEL = dm.n


def lin(**kw):
    """helper not used; linear forms are dicts {dim_index: coeff}"""
    raise NotImplementedError


def cnt_form():
    return {dm[("cand", v)]: 1.0 for v in range(9)}


def add_forms(*forms):
    out = {}
    for f in forms:
        for k, c in f.items():
            out[k] = out.get(k, 0.0) + c
    return out


def scale(f, s):
    return {k: c * s for k, c in f.items()}


def var(name, coeff=1.0):
    return {dm[name]: coeff}


def priority_form():
    # p = 1000 U - 81 cnt - idx   (unique per cell; larger is preferred)
    return add_forms(var("U", 1000.0), scale(cnt_form(), -81.0), var("idx", -1.0))


# ----------------------------------------------------------------------------------------------- MLP builder
class MLP:
    def __init__(self):
        self.units = []          # (form dict, bias)
        self.outs = []           # (unit index, out dim, weight)
        self.b2 = {}

    def unit(self, form, bias):
        self.units.append((dict(form), float(bias)))
        return len(self.units) - 1

    def emit(self, terms, dim, w=1.0):
        """terms: list of (unit, coeff); adds coeff*w*relu(unit) to dim"""
        for u, c in terms:
            self.outs.append((u, dim, c * w))

    # expressions are lists of (unit, coeff)
    def pstep(self, form, bias=0.0):
        """exactly 0 for x<=.25, 1 for x>=.75 where x = form + bias"""
        a = self.unit(scale(form, 2.0), 2.0 * bias - 0.5)
        b = self.unit(scale(form, 2.0), 2.0 * bias - 1.5)
        return [(a, 1.0), (b, -1.0)]

    def AND(self, pos, neg=()):
        f = add_forms(*[var(p) for p in pos], *[var(q, -1.0) for q in neg])
        return self.pstep(f, -(len(pos) - 1))

    def trap(self, form):
        """1 for |x|<=.25, 0 for |x|>=.5"""
        us = [self.unit(form, 0.5), self.unit(form, 0.25), self.unit(form, -0.25), self.unit(form, -0.5)]
        return [(us[0], 4.0), (us[1], -4.0), (us[2], -4.0), (us[3], 4.0)]

    def clear(self, dim):
        """subtract the current value of dim"""
        a = self.unit({dim: 1.0}, 0.0)
        b = self.unit({dim: -1.0}, 0.0)
        self.outs.append((a, dim, -1.0))
        self.outs.append((b, dim, 1.0))

    def set(self, dim, expr_terms):
        self.clear(dim)
        self.emit(expr_terms, dim)

    def arrays(self):
        dff = len(self.units)
        W1 = np.zeros((D_MODEL, dff))
        b1 = np.zeros(dff)
        W2 = np.zeros((dff, D_MODEL))
        b2 = np.zeros(D_MODEL)
        for j, (f, b) in enumerate(self.units):
            for k, c in f.items():
                W1[k, j] += c
            b1[j] = b
        for u, dim, w in self.outs:
            W2[u, dim] += w
        return W1, b1, W2, b2


# ----------------------------------------------------------------------------------------------- layers
layers = []          # dicts: heads (list of spec), Wq, Wk, Wv, Wo, mlp


def layer_L0():
    m = MLP()
    cnt = cnt_form()
    for v in range(9):
        m.set(dm[("solved", v)], m.pstep(add_forms(var(("cand", v), 2.0), scale(cnt, -1.0))))
    m.set(dm["U"], m.pstep(add_forms(cnt, var("is_cell", 20.0)), -21.0))
    m.set(dm["E"], m.pstep(add_forms(var("is_cell"), scale(cnt, -1.0))))
    for v in range(9):
        f = add_forms(var(("cand", v)), *[var(("cand", u), -1.0) for u in range(v)])
        m.set(dm[("g", v)], m.pstep(f))
    return {"heads": [], "mlp": m}


def layer_L1():
    heads = []
    Wq = np.zeros((D_MODEL, 27))
    Wk = np.zeros((D_MODEL, 27))
    Wv = np.zeros((D_MODEL, 54))
    Wo = np.zeros((54, D_MODEL))
    for u, nm in enumerate(["row", "col", "box"]):
        heads.append({"dk": 9, "dv": 18, "mask": "none"})
        for j in range(9):
            Wq[dm[(nm, j)], 9 * u + j] = BETA * 3.0      # sqrt(dk) = 3
            Wk[dm[(nm, j)], 9 * u + j] = 1.0
        for v in range(9):
            Wv[dm[("cand", v)], 18 * u + v] = 1.0
            Wv[dm[("solved", v)], 18 * u + 9 + v] = 1.0
            Wo[18 * u + v, dm[(("C", u), v)]] = 9.0
            Wo[18 * u + 9 + v, dm[("Ssum", v)]] = 9.0
    m = MLP()
    for v in range(9):
        m.set(dm[("elim", v)], m.pstep(add_forms(var(("Ssum", v)), var(("solved", v), -3.0))))
    for v in range(9):
        terms = []
        for u in range(3):
            terms += m.pstep(add_forms(var(("cand", v), 2.0), var((("C", u), v), -1.0)))
        m.set(dm[("Hsum", v)], terms)
    terms = m.pstep(var("E"))
    for u in range(3):
        for v in range(9):
            terms += m.pstep(add_forms(var("is_cell"), var((("C", u), v), -1.0)))
    m.set(dm["contra"], terms)
    for u in range(3):
        for v in range(9):
            m.clear(dm[(("C", u), v)])
    for v in range(9):
        m.clear(dm[("Ssum", v)])
    return {"heads": heads, "Wq": Wq, "Wk": Wk, "Wv": Wv, "Wo": Wo, "mlp": m}


def layer_L2():
    m = MLP()
    for v in range(9):
        f = add_forms(var(("cand", v)), var(("elim", v), -1.0), *[var(("Hsum", w), -1.0) for w in range(9) if w != v])
        m.set(dm[("newc", v)], m.pstep(f))
    return {"heads": [], "mlp": m}


def layer_L3():
    heads = [{"dk": 1, "dv": 3, "mask": "none"}, {"dk": 1, "dv": 1, "mask": "none"}]
    Wq = np.zeros((D_MODEL, 2))
    Wk = np.zeros((D_MODEL, 2))
    Wv = np.zeros((D_MODEL, 4))
    Wo = np.zeros((4, D_MODEL))
    # global uniform over cells
    Wq[dm["one"], 0] = BETA
    Wk[dm["is_cell"], 0] = 1.0
    Wv[dm["contra"], 0] = 1.0
    for v in range(9):
        Wv[dm[("cand", v)], 1] = 1.0
        Wv[dm[("newc", v)], 1] = -1.0
    Wv[dm["U"], 2] = 1.0
    Wo[0, dm["c_con"]] = 81.0
    Wo[1, dm["c_chg"]] = 81.0
    Wo[2, dm["c_u"]] = 81.0
    # selection: argmax of priority
    Wq[dm["one"], 1] = BETA_SEL
    for k, c in priority_form().items():
        Wk[k, 1] += c
        Wv[k, 3] += c
    Wo[3, dm["p_sel"]] = 1.0
    m = MLP()
    m.set(dm["P"], m.pstep(var("c_con")))
    m.set(dm["Q"], m.pstep(add_forms(var("c_chg"), var("c_con", -BIG))))
    m.set(dm["G"], m.pstep(add_forms(var("c_u"), var("c_con", -BIG), var("c_chg", -BIG))))
    m.set(dm["is_sel"], m.trap(add_forms(priority_form(), var("p_sel", -1.0))))
    for nm in ["c_con", "c_chg", "c_u", "p_sel"]:
        m.clear(dm[nm])
    return {"heads": heads, "Wq": Wq, "Wk": Wk, "Wv": Wv, "Wo": Wo, "mlp": m}


def layer_L4():
    m = MLP()
    # candidates
    for v in range(9):
        c = ("cand", v)
        t = []
        for k in range(K):
            t += m.AND(["P", ("D", k + 1), (("S", k), v)])
        t += m.AND(["Q", ("newc", v)])
        t += m.AND([c], ["P", "Q"])
        t += [(u, -w) for u, w in m.AND(["G", "is_sel", c])]
        t += m.AND(["G", "is_sel", ("g", v)])
        m.set(dm[c], t)
    # stack snapshots
    for k in range(K):
        for v in range(9):
            s = (("S", k), v)
            t = m.AND([s], ["G"])
            t += m.AND([s, "G"], [("D", k)])
            t += m.AND(["G", ("D", k), ("cand", v)])
            t += [(u, -w) for u, w in m.AND(["G", ("D", k), "is_sel", ("g", v)])]
            m.set(dm[s], t)
    # depth
    for k in range(K + 1):
        t = m.AND([("D", k)], ["P", "G"])
        if k + 1 <= K:
            t += m.AND([("D", k + 1), "P"])
        if k - 1 >= 0:
            t += m.AND([("D", k - 1), "G"])
        m.set(dm[("D", k)], t)
    return {"heads": [], "mlp": m}


def build():
    layers = [layer_L0(), layer_L1(), layer_L2(), layer_L3(), layer_L4()]
    W = {}
    # embeddings: field 0 = byte value, field 1 = position (clamped)
    E0 = np.zeros((256, D_MODEL))
    E0[ord("0"), [dm[("cand", v)] for v in range(9)]] = 1.0
    for v in range(9):
        E0[ord(str(v + 1)), dm[("cand", v)]] = 1.0
    E1 = np.zeros((NPOS, D_MODEL))
    E1[:, dm["one"]] = 1.0
    E1[:, dm[("D", 0)]] = 1.0
    for r in range(9):
        for c in range(9):
            p = PREFIX + 10 * r + c
            i = 9 * r + c
            E1[p, dm["is_cell"]] = 1.0
            E1[p, dm["idx"]] = float(i)
            E1[p, dm[("row", r)]] = 1.0
            E1[p, dm[("col", c)]] = 1.0
            E1[p, dm[("box", (r // 3) * 3 + c // 3)]] = 1.0
    W["emb.0"], W["emb.1"] = E0, E1
    cfg_layers = []
    for l, L in enumerate(layers):
        W1, b1, W2, b2 = L["mlp"].arrays()
        W[f"L{l}.W1"], W[f"L{l}.b1"], W[f"L{l}.W2"], W[f"L{l}.b2"] = W1, b1, W2, b2
        if L["heads"]:
            for nm in ["Wq", "Wk", "Wv", "Wo"]:
                W[f"L{l}.{nm}"] = L[nm]
        cfg_layers.append({"heads": L["heads"], "d_ff": W1.shape[1]})
    alphabet = [""] + [str(v) for v in range(1, 10)]
    Wout = np.zeros((D_MODEL, 10))
    bout = np.zeros(10)
    Wout[dm["is_cell"], 0] = -10.0
    bout[0] = 9.5
    for v in range(9):
        Wout[dm[("cand", v)], v + 1] = 1.0
    W["out.W"], W["out.b"] = Wout, bout
    cfg = {"d_model": D_MODEL, "fields": ["byte", "position"], "layers": cfg_layers,
           "loops": {"const": R_LOOPS, "lin": 0, "div": 1, "log2": 0}, "alphabet": alphabet}
    return cfg, W


def save(cfg, W, out=OUT):
    os.makedirs(out, exist_ok=True)
    json.dump(cfg, open(os.path.join(out, "network.json"), "w"), indent=1)
    store = {}
    for k, a in W.items():
        nz = np.flatnonzero(a)
        if nz.size < a.size / 3:          # sparse storage
            store[k + ".shape"] = np.array(a.shape, dtype=np.int64)
            store[k + ".idx"] = nz.astype(np.int64)
            store[k + ".val"] = a.ravel()[nz].astype(np.float64)
        else:
            store[k] = a.astype(np.float64)
    np.savez_compressed(os.path.join(out, "weights.npz"), **store)
    shutil.copy(os.path.join(HERE, "tokenizer.py"), os.path.join(out, "tokenizer.py"))


if __name__ == "__main__":
    cfg, W = build()
    save(cfg, W)
    tot = sum(a.size for a in W.values())
    nz = sum(int(np.count_nonzero(a)) for a in W.values())
    print(f"d_model {D_MODEL}, d_ff {[L['d_ff'] for L in cfg['layers']]}, params {tot:,}, nonzero {nz:,}")
