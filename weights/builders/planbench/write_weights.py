"""Writes every weight of the three PlanBench plan-generation networks (no training, no data).

    python3 -I work/write_weights.py

Each network is ONE transformer layer (attention heads + one MLP) applied R times. Every quantity in the residual
stream is written with "set" semantics (the MLP adds new - old), and every attention head writes into a scratch
group that the same layer's MLP consumes and clears, so looping the layer is idempotent: each loop resolves one more
step of the dependency chain (characters -> local patterns -> facts -> relations -> heights / lookups -> output).

Tokenizer (generic): one token per UTF-8 byte, fields (byte value, position clipped to P-1).
Output: an 'emit' readout; a fixed region of positions (the "canvas", positions 1..K) carries learned positional
features that say which piece of which plan step it writes; all other positions write "".
"""
import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
P = 16384                    # position table size (prompt positions are clipped to P - 1 by the tokenizer)
NFREQ = 14                   # sinusoid frequencies 2*pi/2^(k+1), k < 14: unique offsets for |delta| < 2^14
CHARS = "abcdefghijklmnopqrstuvwxyz0123456789 ,.\n[]_"
NCH = len(CHARS) + 2         # + "other" + "upper-case flag"
OTHER, UPPER = len(CHARS), len(CHARS) + 1

TOKENIZER = '''"""Generic byte tokenizer: one token per UTF-8 byte of the text, two fields:
field 0 = the byte value (0..255), field 1 = the token's position, clipped to %d."""
P = %d


def tokenize(text: str):
    data = text.encode("utf-8")
    return [(b, min(i, P - 1)) for i, b in enumerate(data)]
''' % (P - 1, P)


def char_index(byte):
    """(class index, upper flag) of a byte."""
    c = chr(byte) if byte < 128 else None
    up = 0
    if c is not None and c.isupper():
        c, up = c.lower(), 1
    return (CHARS.index(c) if (c is not None and c in CHARS) else OTHER), up


class Net:
    def __init__(self):
        self.groups = {}
        self.d = 0
        self.heads = []
        self.units = []
        self.pos_feats = {}      # pos -> list of (dim, val)

    # ---------------------------------------------------------------- residual layout
    def g(self, name, n=1):
        assert name not in self.groups, name
        self.groups[name] = list(range(self.d, self.d + n))
        self.d += n
        return self.groups[name]

    def posf(self, p, dim, val=1.0):
        self.pos_feats.setdefault(p, []).append((dim, val))

    # ---------------------------------------------------------------- MLP units
    def unit(self, inp, bias, out):
        """One hidden ReLU unit: relu(sum w*h[dim] + bias), added to h[dim'] * w'."""
        self.units.append((list(inp), float(bias), list(out)))

    def clear(self, dims):
        for d in dims:
            self.unit([(d, 1.0)], 0.0, [(d, -1.0)])
            self.unit([(d, -1.0)], 0.0, [(d, 1.0)])

    def set_relus(self, dst, terms):
        """h[dst] <- sum coef * relu(inp + bias) (the old value is cancelled exactly)."""
        self.clear([dst])
        for inp, bias, coef in terms:
            self.unit(inp, bias, [(dst, coef)])

    def set_lin(self, dst, inp, bias=0.0):
        """h[dst] <- inp + bias (linear), with 2 units: delta = (inp + bias - old)."""
        a = list(inp) + [(dst, -1.0)]
        self.unit(a, bias, [(dst, 1.0)])
        self.unit([(d, -w) for d, w in a], -bias, [(dst, -1.0)])

    # ---------------------------------------------------------------- attention heads
    def head(self, q, k, v, dk, dv, mask, out):
        """q, k: lists (dim, col, w) for the (d, dk) projections (scores = q.k, the 1/sqrt(dk) of the runner is
        compensated here); v: (dim, col, w) for (d, dv); out: (col, dim, w) for (dv, d)."""
        s = math.sqrt(dk)
        self.heads.append(dict(q=[(a, b, w * s) for a, b, w in q], k=k, v=v, o=out, dk=dk, dv=dv, mask=mask))

    def offset_head(self, off, src, scratch, A=16.0):
        """scratch[i] <- h[src[i]] at position p + off (sinusoidal relative addressing, peak gap 2A)."""
        sn = self.groups["sin"]
        q, k = [], []
        for f in range(NFREQ):
            w = 2 * math.pi / 2 ** (f + 1)
            c, s = math.cos(w * off), math.sin(w * off)
            C, S = sn[2 * f], sn[2 * f + 1]
            q += [(C, 2 * f, A * c), (S, 2 * f, -A * s), (C, 2 * f + 1, A * s), (S, 2 * f + 1, A * c)]
            k += [(C, 2 * f, 1.0), (S, 2 * f + 1, 1.0)]
        v = [(d, i, 1.0) for i, d in enumerate(src)]
        o = [(i, d, 1.0) for i, d in enumerate(scratch)]
        self.head(q, k, v, 2 * NFREQ, len(src), "none", o)

    def recency_head(self, cond, val, scratch, lam=1.0, M=40000.0, bos=None):
        """Causal: the most recent position j <= p with cond(j)=1 (cond: list (dim, w)); scratch <- val there.
        bos: score of the first position (a default) if given."""
        one, pos, b = self.groups["one"][0], self.groups["pos"][0], self.groups["bos"][0]
        q = [(one, 0, M), (one, 1, lam)]
        k = [(d, 0, w) for d, w in cond] + [(pos, 1, 1.0)]
        if bos is not None:
            q.append((one, 2, bos))
            k.append((b, 2, 1.0))
        v = [(d, i, 1.0) for i, d in enumerate(val)]
        o = [(i, d, 1.0) for i, d in enumerate(scratch)]
        self.head(q, k, v, 3, len(val), "causal", o)

    def match_head(self, pairs, val, scratch, A=40.0, bos=None, mask="none", extra_q=(), extra_k=(), vsum=None):
        """Content lookup. pairs: list of (query dims, key dims), the score is A * sum_i q_i k_i; a default first
        position gets score `bos`. val: value dims (vsum: list of lists of dims summed into each value col)."""
        q, k, col = [], [], 0
        for qd, kd in pairs:
            for a, b in zip(qd, kd):
                q.append((a, col, A))
                k.append((b, col, 1.0))
                col += 1
        one, bo = self.groups["one"][0], self.groups["bos"][0]
        if bos is not None:
            q.append((one, col, bos))
            k.append((bo, col, 1.0))
            col += 1
        for (a, w), (b, w2) in zip(extra_q, extra_k):
            q.append((a, col, w))
            k.append((b, col, w2))
            col += 1
        if vsum is None:
            vsum = [[d] for d in val]
        v = [(d, i, 1.0) for i, ds in enumerate(vsum) for d in ds]
        o = [(i, d, 1.0) for i, d in enumerate(scratch)]
        self.head(q, k, v, col, len(vsum), mask, o)

    # ---------------------------------------------------------------- base features
    def base(self):
        self.g("ch", NCH)
        self.g("one")
        self.g("sin", 2 * NFREQ)
        self.g("pos")
        self.g("bos")

    def ch(self, c):
        return self.groups["ch"][CHARS.index(c)]

    # ---------------------------------------------------------------- write out
    def save(self, out_dir, alphabet, logit_dims, loops, empty_bias=0.5):
        os.makedirs(out_dir, exist_ok=True)
        d = self.d
        E0 = np.zeros((256, d))
        chd, one = self.groups["ch"], self.groups["one"][0]
        for b in range(256):
            ci, up = char_index(b)
            E0[b, chd[ci]] = 1.0
            if up:
                E0[b, chd[UPPER]] = 1.0
            E0[b, one] = 1.0
        rows, cols, vals = [], [], []
        sn, pd, bo = self.groups["sin"], self.groups["pos"][0], self.groups["bos"][0]
        for p in range(P):
            for f in range(NFREQ):
                w = 2 * math.pi / 2 ** (f + 1)
                rows += [p, p]
                cols += [sn[2 * f], sn[2 * f + 1]]
                vals += [math.cos(w * p), math.sin(w * p)]
            if p:
                rows.append(p); cols.append(pd); vals.append(float(p))
            if p == 0:
                rows.append(p); cols.append(bo); vals.append(1.0)
            for dim, v in self.pos_feats.get(p, []):
                rows.append(p); cols.append(dim); vals.append(v)
        rows, cols, vals = np.array(rows), np.array(cols), np.array(vals)
        W = {}

        def sparse(name, shape, r, c, v):
            flat = np.asarray(r, dtype=np.int64) * shape[1] + np.asarray(c, dtype=np.int64)
            acc = {}
            for f, x in zip(flat.tolist(), np.asarray(v, dtype=np.float64).tolist()):
                acc[f] = acc.get(f, 0.0) + x
            keys = np.array(sorted(k for k in acc if acc[k] != 0.0), dtype=np.int64)
            W[name + ".shape"] = np.array(shape, dtype=np.int64)
            W[name + ".idx"] = keys
            W[name + ".val"] = np.array([acc[k] for k in keys.tolist()], dtype=np.float64)

        sparse("emb.0", (256, d), *np.nonzero(E0), E0[np.nonzero(E0)])
        sparse("emb.1", (P, d), rows, cols, vals)
        # the one layer
        heads, sk, sv = self.heads, sum(h["dk"] for h in self.heads), sum(h["dv"] for h in self.heads)
        Q, K, V, O = ([], [], []), ([], [], []), ([], [], []), ([], [], [])
        qo = vo = 0
        for h in heads:
            for a, b, w in h["q"]:
                Q[0].append(a); Q[1].append(qo + b); Q[2].append(w)
            for a, b, w in h["k"]:
                K[0].append(a); K[1].append(qo + b); K[2].append(w)
            for a, b, w in h["v"]:
                V[0].append(a); V[1].append(vo + b); V[2].append(w)
            for a, b, w in h["o"]:
                O[0].append(vo + a); O[1].append(b); O[2].append(w)
            qo += h["dk"]
            vo += h["dv"]
        sparse("L0.Wq", (d, sk), *Q)
        sparse("L0.Wk", (d, sk), *K)
        sparse("L0.Wv", (d, sv), *V)
        sparse("L0.Wo", (sv, d), *O)
        nu = len(self.units)
        W1, W2, b1 = ([], [], []), ([], [], []), np.zeros(nu)
        for i, (inp, bias, out) in enumerate(self.units):
            for a, w in inp:
                W1[0].append(a); W1[1].append(i); W1[2].append(w)
            b1[i] = bias
            for a, w in out:
                W2[0].append(i); W2[1].append(a); W2[2].append(w)
        sparse("L0.W1", (d, nu), *W1)
        W["L0.b1"] = b1
        sparse("L0.W2", (nu, d), *W2)
        W["L0.b2"] = np.zeros(d)
        Wout = np.zeros((d, len(alphabet)))
        bout = np.zeros(len(alphabet))
        bout[alphabet.index("")] = empty_bias
        for i, s in enumerate(alphabet):
            if s != "":
                Wout[logit_dims[s], i] = 1.0
        sparse("out.W", (d, len(alphabet)), *np.nonzero(Wout), Wout[np.nonzero(Wout)])
        W["out.b"] = bout
        np.savez(os.path.join(out_dir, "weights.npz"), **W)
        cfg = {"d_model": d, "fields": ["byte", "position"],
               "layers": [{"heads": [{"dk": h["dk"], "dv": h["dv"], "mask": h["mask"]} for h in heads],
                           "d_ff": nu}],
               "loops": loops, "alphabet": alphabet}
        json.dump(cfg, open(os.path.join(out_dir, "network.json"), "w"), indent=1)
        open(os.path.join(out_dir, "tokenizer.py"), "w").write(TOKENIZER)
        nz = sum(int(np.count_nonzero(W[k])) for k in W if k.endswith(".val")) + \
            sum(int(np.count_nonzero(W[k])) for k in W if not (k.endswith(".val") or k.endswith(".idx")
                                                               or k.endswith(".shape")))
        tot = 256 * d + P * d + d * (2 * sk + sv) + sv * d + 2 * d * nu + nu + d + d * len(alphabet) + len(alphabet)
        return dict(d_model=d, heads=len(heads), d_ff=nu, params_total=tot, params_nonzero=nz)


# ==================================================================== shared helpers
def pattern(net, spec):
    """(inputs, bias) of a ReLU that is 1 iff every (source dims, chars) in spec matches, else <= 0.
    spec: list of (dims of a char vector at some offset, chars allowed) ."""
    inp = []
    for vec, chars in spec:
        for c in chars:
            if c == "UP":
                inp.append((vec[UPPER], 1.0))
            else:
                inp.append((vec[CHARS.index(c)], 1.0))
    return inp, -(len(spec) - 1)


def char_offsets(net, offsets, name="T"):
    """Offset heads reading the whole char vector; returns {offset: dims} (offset 0 = the embedding itself)."""
    res = {0: net.groups["ch"]}
    for o in offsets:
        t = net.g(f"T_ch{o}", NCH)
        net.offset_head(o, net.groups["ch"], t)
        res[o] = t
    return res


def markers(net, W):
    """Dests PE ("D]" of [PLAN END]), INIT ("itial"), GOAL ("goal ") at their positions, from char windows W."""
    out = {}
    for name, spec in (("PE", [(W[-1], "d"), (W[-1], ["UP"]), (W[0], "]")]),
                       ("INIT", [(W[-2], "i"), (W[-1], "t"), (W[0], "i"), (W[1], "a"), (W[2], "l")]),
                       ("GOAL", [(W[-2], "g"), (W[-1], "o"), (W[0], "a"), (W[1], "l"), (W[2], " ")])):
        d = net.g(name)[0]
        inp, b = pattern(net, spec)
        net.set_relus(d, [(inp, b, 1.0)])
        out[name] = d
    return out


def region_heads(net, mk):
    """Scratch aPE (1 if after [PLAN END]) and gR (1 if the most recent INIT/GOAL marker is GOAL)."""
    aPE = net.g("T_aPE")
    net.recency_head([(mk["PE"], 1.0)], [mk["PE"]], aPE, lam=0.0, M=40.0, bos=20.0)
    gR = net.g("T_gR")
    net.recency_head([(mk["INIT"], 1.0), (mk["GOAL"], 1.0)], [mk["GOAL"]], gR, lam=1.0, M=40000.0)
    return aPE[0], gR[0]


def write_emit(net, alphabet_strings):
    """Logit dims for every non-empty alphabet entry."""
    return {s: net.g("LG_" + repr(s))[0] for s in alphabet_strings}


# ==================================================================== Blocksworld family
COLORS = ["red", "blue", "orange", "yellow", "white", "magenta", "black", "cyan", "green", "violet", "silver",
          "gold"]
NB = 12          # object identities
LMAX = 6         # tower levels handled (towers of up to 7 blocks)


def build_bw(variant, out_dir):
    net = Net()
    net.base()
    if variant == "blocksworld":
        names = [c + " block" for c in COLORS]
        P1 = ["unstack the ", "X", " from on top of the ", "Y", "\nput down the ", "X", "\n"]
        P2 = ["pick up the ", "X", "\nstack the ", "X", " on top of the ", "Y", "\n"]
    else:
        names = ["object " + chr(97 + i) for i in range(NB)]
        P1 = ["feast ", "X", " from ", "Y", "\nsuccumb ", "X", "\n"]
        P2 = ["attack ", "X", "\novercome ", "X", " from ", "Y", "\n"]
    fixed = []
    for s in P1 + P2:
        if s not in ("X", "Y") and s not in fixed:
            fixed.append(s)
    alphabet = [""] + fixed + names
    LG = write_emit(net, fixed + names)

    # ---- canvas positional features
    Q1, Q2 = net.g("Q1", NB), net.g("Q2", NB)
    LV = net.g("LV", LMAX)
    FX = {s: net.g("FX_" + repr(s))[0] for s in fixed}
    XN = net.g("XN", NB)
    YN = net.g("YN")[0]
    p = 1
    for phase, pieces, levels in ((1, P1, range(LMAX, 0, -1)), (2, P2, range(1, LMAX + 1))):
        for lev in levels:
            for b in range(NB):
                for piece in pieces:
                    net.posf(p, (Q1 if phase == 1 else Q2)[b])
                    net.posf(p, LV[lev - 1])
                    if piece == "X":
                        net.posf(p, XN[b])
                    elif piece == "Y":
                        net.posf(p, YN)
                    else:
                        net.posf(p, FX[piece])
                    p += 1
    canvas_end = p

    # ---- loop step 1: char windows -> local patterns
    W = char_offsets(net, [-2, -1, 1, 2])
    mk = markers(net, W)
    if variant == "blocksworld":
        CE, CS = net.g("CE", NB), net.g("CS", NB)       # color word ends / starts here
        for i, c in enumerate(COLORS):
            inp, b = pattern(net, [(W[-1], c[-2]), (W[0], c[-1])])
            net.set_relus(CE[i], [(inp, b, 1.0)])
            inp, b = pattern(net, [(W[0], c[0]), (W[1], c[1]), (W[2], c[2])])
            net.set_relus(CS[i], [(inp, b, 1.0)])
        F = net.g("TOP")[0]                              # the 'f' of "... on top of the ..."
        inp, b = pattern(net, [(W[-2], " "), (W[-1], "o"), (W[0], "f"), (W[1], " "), (W[2], "t")])
        net.set_relus(F, [(inp, b, 1.0)])
        Fval = (F, 1.0)
        srcX, offX, srcY, offY = CE, -19, CS, 6
    else:
        F = net.g("CRAVE")[0]                            # the 'v' of "craves"
        inp, b = pattern(net, [(W[-2], "r"), (W[-1], "a"), (W[0], "v"), (W[1], "e"), (W[2], "s")])
        net.set_relus(F, [(inp, b, 1.0)])
        letters = [net.ch(chr(97 + i)) for i in range(NB)]
        TF = net.g("T_F")
        net.offset_head(-11, [F], TF)
        Fval = (TF[0], 1.0)
        srcX, offX, srcY, offY = letters, -16, letters, 0

    # ---- loop step 2: facts "X on Y" of the problem's initial state (I_) and goal (G_)
    aPE, gR = region_heads(net, mk)
    TX = net.g("T_X", NB)
    net.offset_head(offX, srcX, TX)
    if offY == 0:
        TY = srcY
    else:
        TY = net.g("T_Y", NB)
        net.offset_head(offY, srcY, TY)
    IX, IY, GX, GY = (net.g(n, NB) for n in ("IX", "IY", "GX", "GY"))
    AI, AG = net.g("AI")[0], net.g("AG")[0]
    for c in range(NB):
        net.set_relus(IX[c], [([Fval, (aPE, 1), (gR, -1), (TX[c], 1)], -2, 1.0)])
        net.set_relus(IY[c], [([Fval, (aPE, 1), (gR, -1), (TY[c], 1)], -2, 1.0)])
        net.set_relus(GX[c], [([Fval, (aPE, 1), (gR, 1), (TX[c], 1)], -3, 1.0)])
        net.set_relus(GY[c], [([Fval, (aPE, 1), (gR, 1), (TY[c], 1)], -3, 1.0)])
    net.set_relus(AI, [([Fval, (aPE, 1), (gR, -1)], -1, 1.0)])
    net.set_relus(AG, [([Fval, (aPE, 1), (gR, 1)], -2, 1.0)])

    # ---- loop steps 3..: heights. hv(fact X on Y) = 1 + hv(fact Y on Z) (0 if none): init and goal separately.
    # The same lookup carries "well placed" (WP): an initial fact X on Y is WP iff the goal also has X on Y and
    # either the fact below (Y on Z) is WP or Y is a base (on the table at the start, no goal fact about Y).
    HV, WP = net.g("HV")[0], net.g("WP")[0]
    TH = net.g("T_H", 3)
    net.match_head([(IY, IX), (GY, GX)], None, TH, A=40.0, bos=20.0, vsum=[[HV], [WP], [AI, AG]])
    TGM = net.g("T_GM")                       # does the goal hold this initial fact?
    net.match_head([(IX, GX), (IY, GY)], [AG], TGM, A=40.0, bos=60.0)
    TFG = net.g("T_FG")                       # init anchor: does Y have a goal fact; goal anchor: does X stay?
    net.match_head([(IY, GX), (GX, IX)], None, TFG, A=40.0, bos=20.0, vsum=[[AG, WP]])
    TFX = net.g("T_FX")                       # init anchor: does X have a goal fact
    net.match_head([(IX, GX)], [AG], TFX, A=40.0, bos=20.0)
    TTY = net.g("T_TY")                       # init anchor: is Y the support of some goal fact
    net.match_head([(IY, GY)], [AG], TTY, A=40.0, bos=20.0)
    FI, FGI, GWP, BASE = (net.g(n)[0] for n in ("FI", "FGI", "GWP", "BASE"))
    net.set_lin(FI, [(TH[2], 1.0)])
    net.set_relus(FGI, [([(TFG[0], 1.0), (AI, 1.0)], -1.0, 1.0)])
    net.set_relus(GWP, [([(TFG[0], 1.0), (AG, 1.0)], -1.0, 1.0)])
    net.set_relus(BASE, [([(FI, -1.0), (FGI, -1.0)], 1.0, 1.0)])
    BIG = 64.0
    # stay(X on Y) = [goal X on Y, or no goal for X] and [Y is no other block's goal support] and
    #                [Y stays, or Y is a base]  ->  relu(2 GM - FX - TY + stay(below) + base)
    net.set_relus(WP, [([(TGM[0], 2.0), (TFX[0], -1.0), (TTY[0], -1.0), (TH[1], 1.0), (BASE, 1.0), (AI, BIG)],
                        -BIG, 1.0)])
    x = [(TH[0], 1.0)]
    gate = [(AI, BIG), (AG, BIG)]
    net.set_relus(HV, [(x + gate, 1.0 - BIG, 1.0)])              # hv <- (T + 1) if anchor else 0
    HK = net.g("HK", LMAX)                                         # one-hot of hv (anchors only)
    for k in range(1, LMAX + 1):
        net.set_relus(HK[k - 1], [(x + gate, 1.0 - k + 1 - BIG, 1.0), (x + gate, 1.0 - k - BIG, -2.0),
                                  (x + gate, 1.0 - k - 1 - BIG, 1.0)])

    # ---- canvas: look up the fact about block b (initial for phase 1, goal for phase 2); WP blocks stay
    TC = net.g("T_C", LMAX + NB + 1)
    net.match_head([(Q1, IX), (Q2, GX)], None, TC, A=40.0, bos=20.0,
                   vsum=[[HK[k]] for k in range(LMAX)] + [[IY[c], GY[c]] for c in range(NB)] + [[WP, GWP]])
    CL, CY, CW = net.g("CL", LMAX), net.g("CY", NB), net.g("CW")[0]
    for k in range(LMAX):
        net.set_lin(CL[k], [(TC[k], 1.0)])
    for c in range(NB):
        net.set_lin(CY[c], [(TC[LMAX + c], 1.0)])
    net.set_lin(CW, [(TC[LMAX + NB], 1.0)])
    ACT = net.g("ACT")[0]
    net.set_relus(ACT, [([(CL[k], 1.0), (LV[k], 1.0), (CW, -1.0)], -1.0, 1.0) for k in range(LMAX)])
    for s in fixed:
        net.set_relus(LG[s], [([(FX[s], 1.0), (ACT, 1.0)], -1.0, 1.0)])
    for b, nm in enumerate(names):
        net.set_relus(LG[nm], [([(XN[b], 1.0), (ACT, 1.0)], -1.0, 1.0),
                               ([(YN, 1.0), (CY[b], 1.0), (ACT, 1.0)], -2.0, 1.0)])

    for gname, dims in list(net.groups.items()):      # every scratch group is consumed above, then cleared
        if gname.startswith("T_"):
            net.clear(dims)
    loops = {"const": 2 + LMAX + 8, "lin": 0, "div": 1, "log2": 0}
    info = net.save(out_dir, alphabet, LG, loops)
    info["canvas_positions"] = canvas_end - 1
    return info


# ==================================================================== Logistics
ND = 10     # digits / indices 0..9
NCAN_P = 10   # package slots


def build_logistics(out_dir):
    net = Net()
    net.base()
    dig = [net.ch(str(i)) for i in range(ND)]

    # ---- plan template, per package p with a goal: initial place S (city X), goal G (city Y), airports AX, AY,
    # trucks tX, tY, one airplane aK. "F" is the vehicle's current place: the destination of its latest earlier
    # required move, else its initial place. Each action is 13 pieces (fixed strings or digit sources) plus a
    # silent 14th position that carries the action's move as a key for later actions.
    NP = 14

    def act(*parts):
        parts = list(parts) + ["\n"]
        assert len(parts) <= NP - 1, parts
        return parts + [None] * (NP - len(parts))

    def drive(t, a, b, c):
        return act("drive truck_", t, " from location_", a + "I", "_", a + "J", " to location_", b + "I", "_",
                   b + "J", " in city_", c)

    def load(p, kind, v, l):
        return act("load package_", p, f" into {kind}_", v, " at location_", l + "I", "_", l + "J")

    def unload(p, kind, v, l):
        return act("unload package_", p, f" from {kind}_", v, " at location_", l + "I", "_", l + "J")

    def fly(v, a, b):
        return act("fly airplane_", v, " from location_", a + "I", "_", a + "J", " to location_", b + "I", "_",
                   b + "J")

    # (pieces, required-group, move (vehicle, destination) or None, dynamic from?)
    # groups: 1 = same city or not at its airport, 2 = other city and not at its airport, 3 = same city,
    #         4 = other city, 5 = other city and goal not the airport; all also need: goal given, not there yet
    pkg_slot = [
        (drive("tX", "F", "S", "X"), 1, ("tX", "S"), True),
        (load("#p", "truck", "tX", "S"), 1, None, False),
        (drive("tX", "S", "AX", "X"), 2, ("tX", "AX"), False),
        (drive("tX", "S", "G", "X"), 3, ("tX", "G"), False),
        (unload("#p", "truck", "tX", "AX"), 2, None, False),
        (unload("#p", "truck", "tX", "G"), 3, None, False),
        (fly("aK", "F", "AX"), 4, ("pl", "AX"), True),
        (load("#p", "airplane", "aK", "AX"), 4, None, False),
        (fly("aK", "AX", "AY"), 4, ("pl", "AY"), False),
        (unload("#p", "airplane", "aK", "AY"), 4, None, False),
        (drive("tY", "F", "AY", "Y"), 5, ("tY", "AY"), True),
        (load("#p", "truck", "tY", "AY"), 5, None, False),
        (drive("tY", "AY", "G", "Y"), 5, ("tY", "G"), False),
        (unload("#p", "truck", "tY", "G"), 5, None, False)]
    SRC = ["tX", "tY", "aK", "SI", "SJ", "GI", "GJ", "AXI", "AXJ", "AYI", "AYJ", "FI", "FJ", "X", "Y"]
    fixed = []
    for a, _, _, _ in pkg_slot:
        for s in a:
            if s is not None and s not in SRC and s != "#p" and s not in fixed:
                fixed.append(s)
    digits = [str(i) for i in range(ND)]
    alphabet = [""] + fixed + digits
    LG = write_emit(net, fixed + digits)

    # canvas positional features
    QP = net.g("QP", ND)
    FX = {s: net.g("FX_" + repr(s))[0] for s in fixed}
    SEL = {s: net.g("SEL_" + s)[0] for s in SRC}
    PD = net.g("PD", ND)                     # digit known from the position (package p)
    FGr = {g: net.g(f"FG{g}")[0] for g in range(1, 6)}
    DYN = net.g("DYN")[0]
    SQ = {v: net.g("SQ_" + v)[0] for v in ("tX", "tY", "pl")}            # dynamic action: whose place
    DD = {d: net.g("DD_" + d)[0] for d in ("S", "AX", "AY")}             # dynamic action: its destination
    KEY = {v: net.g("KEY_" + v)[0] for v in ("tX", "tY", "pl")}          # key slot of a move: vehicle
    KD = {d: net.g("KD_" + d)[0] for d in ("S", "AX", "G", "AY")}        # key slot of a move: destination
    p = 1
    for k in range(NCAN_P):
        for pieces, grp, move, dyn in pkg_slot:
            for i, piece in enumerate(pieces):
                net.posf(p, QP[k])
                net.posf(p, FGr[grp])
                if dyn:
                    net.posf(p, DYN)
                    net.posf(p, SQ[move[0]])
                    net.posf(p, DD[move[1]])
                if i == NP - 1 and move is not None:
                    net.posf(p, KEY[move[0]])
                    net.posf(p, KD[move[1]])
                if piece is None:
                    pass
                elif piece == "#p":
                    net.posf(p, PD[k])
                elif piece in SRC:
                    net.posf(p, SEL[piece])
                else:
                    net.posf(p, FX[piece])
                p += 1
    canvas_end = p

    # ---- step 1: local patterns from a char window
    W = char_offsets(net, [-4, -3, -2, -1, 1, 2, 3, 5, 6])
    mk = markers(net, W)
    alld = "0123456789"
    sub = [(W[1], " "), (W[2], "i"), (W[3], "s")]          # "<name> is ..."
    pats = {
        "LOC_AN": [(W[0], alld), (W[-1], "_"), (W[-2], alld), (W[-3], "_"), (W[-4], "n")] + sub
                  + [(W[5], "a"), (W[6], "n")],
        "LOC_IN": [(W[0], alld), (W[-1], "_"), (W[-2], alld), (W[-3], "_"), (W[-4], "n")] + sub
                  + [(W[5], "i"), (W[6], "n")],
        "TRK_AT": [(W[0], alld), (W[-1], "_"), (W[-2], "k")] + sub + [(W[5], "a"), (W[6], "t")],
        "PLN_AT": [(W[0], alld), (W[-1], "_"), (W[-2], "e"), (W[-3], "n")] + sub + [(W[5], "a"), (W[6], "t")],
        "PKG_AT": [(W[0], alld), (W[-1], "_"), (W[-2], "e"), (W[-3], "g")] + sub + [(W[5], "a"), (W[6], "t")],
    }
    PF = {}
    for name, spec in pats.items():
        PF[name] = net.g(name)[0]
        inp, b = pattern(net, spec)
        net.set_relus(PF[name], [(inp, b, 1.0)])

    # ---- step 2: facts of the problem (initial: aPE & !gR, goal: aPE & gR)
    aPE, gR = region_heads(net, mk)
    D = {}
    for o in (-2, 17, 19, 22):
        D[o] = net.g(f"T_d{o}", ND)
        net.offset_head(o, dig, D[o])
    D[0] = dig

    def fact(name, flag, goal, digsrc):
        dims = net.g(name, ND)
        for i in range(ND):
            if goal:
                net.set_relus(dims[i], [([(flag, 1), (aPE, 1), (gR, 1), (digsrc[i], 1)], -3, 1.0)])
            else:
                net.set_relus(dims[i], [([(flag, 1), (aPE, 1), (gR, -1), (digsrc[i], 1)], -2, 1.0)])
        return dims

    def flagd(name, flags, goal):
        d = net.g(name)[0]
        terms = []
        for f in flags:
            if goal:
                terms.append(([(f, 1), (aPE, 1), (gR, 1)], -2, 1.0))
            else:
                terms.append(([(f, 1), (aPE, 1), (gR, -1)], -1, 1.0))
        net.set_relus(d, terms)
        return d

    AP_I, AP_J = fact("AP_I", PF["LOC_AN"], 0, D[-2]), fact("AP_J", PF["LOC_AN"], 0, D[0])
    AP_F = flagd("AP_F", [PF["LOC_AN"]], 0)
    IC_I, IC_J, IC_C = (fact("IC_I", PF["LOC_IN"], 0, D[-2]), fact("IC_J", PF["LOC_IN"], 0, D[0]),
                        fact("IC_C", PF["LOC_IN"], 0, D[22]))
    TK_K, TK_I, TK_J = (fact("TK_K", PF["TRK_AT"], 0, D[0]), fact("TK_I", PF["TRK_AT"], 0, D[17]),
                        fact("TK_J", PF["TRK_AT"], 0, D[19]))
    TK_F = flagd("TK_F", [PF["TRK_AT"]], 0)
    PL_K, PL_I, PL_J = (fact("PL_K", PF["PLN_AT"], 0, D[0]), fact("PL_I", PF["PLN_AT"], 0, D[17]),
                        fact("PL_J", PF["PLN_AT"], 0, D[19]))
    PL_F = flagd("PL_F", [PF["PLN_AT"]], 0)
    PK_K, PK_I, PK_J = (fact("PK_K", PF["PKG_AT"], 0, D[0]), fact("PK_I", PF["PKG_AT"], 0, D[17]),
                        fact("PK_J", PF["PKG_AT"], 0, D[19]))
    GL_K, GL_I, GL_J = (fact("GL_K", PF["PKG_AT"], 1, D[0]), fact("GL_I", PF["PKG_AT"], 1, D[17]),
                        fact("GL_J", PF["PKG_AT"], 1, D[19]))
    GL_F = flagd("GL_F", [PF["PKG_AT"]], 1)

    # ---- step 3: city of every airport fact and of every truck's location
    T_c = net.g("T_c", ND)
    net.match_head([([*AP_I, *TK_I], [*IC_I, *IC_I]), ([*AP_J, *TK_J], [*IC_J, *IC_J])], IC_C, T_c, bos=60.0)
    AP_C, TK_C = net.g("AP_C", ND), net.g("TK_C", ND)
    for i in range(ND):
        net.set_relus(AP_C[i], [([(T_c[i], 1), (AP_F, 1)], -1, 1.0)])
        net.set_relus(TK_C[i], [([(T_c[i], 1), (TK_F, 1)], -1, 1.0)])

    # ---- canvas lookups (each resolves once its query is ready)
    def lookup(name, pairs, vals, bos, **kw):
        T = net.g("T_" + name, len(vals) if vals is not None else len(kw["vsum"]))
        net.match_head(pairs, vals, T, bos=bos, **kw)
        dests = net.g(name, len(T))
        for t, dd in zip(T, dests):
            net.set_lin(dd, [(t, 1.0)])
        return dests

    one, posd = net.groups["one"][0], net.groups["pos"][0]
    G = lookup("G", [(QP, GL_K)], [*GL_I, *GL_J, GL_F], 20.0)                # goal place of package p
    S = lookup("S", [(QP, PK_K)], [*PK_I, *PK_J], 20.0)                     # initial place of package p
    SI, SJ = S[:ND], S[ND:]
    GI, GJ, GF = G[:ND], G[ND:2 * ND], G[2 * ND]
    # an airplane (the last one listed) and where it starts
    H = lookup("H", [], [*PL_K, *PL_I, *PL_J], None,
               extra_q=[(one, 1.0), (one, 1.0)], extra_k=[(PL_F, 40000.0), (posd, 1.0)])
    aK, HI, HJ = H[:ND], H[ND:2 * ND], H[2 * ND:]
    X = lookup("X", [(SI, IC_I), (SJ, IC_J)], IC_C, 60.0)                     # city of S
    Y = lookup("Y", [(GI, IC_I), (GJ, IC_J)], IC_C, 60.0)                     # city of G
    AX = lookup("AX", [(X, AP_C)], [*AP_I, *AP_J], 20.0)                      # airport of X
    AY = lookup("AY", [(Y, AP_C)], [*AP_I, *AP_J], 20.0)
    tX = lookup("tX", [(X, TK_C)], TK_K, 20.0)                                # truck of X
    tY = lookup("tY", [(Y, TK_C)], TK_K, 20.0)
    TXL = lookup("TXL", [(tX, TK_K)], [*TK_I, *TK_J], 20.0)                   # where truck tX starts
    TYL = lookup("TYL", [(tY, TK_K)], [*TK_I, *TK_J], 20.0)

    def eq_pair(name, a, b):
        e = [net.g(f"e{name}{i}")[0] for i in range(len(a))]
        for ed, (ga, gb) in zip(e, zip(a, b)):
            net.set_relus(ed, [([(ga[c], 1.0), (gb[c], 1.0)], -1.0, 1.0) for c in range(ND)])
        d = net.g("EQ" + name)[0]
        net.set_relus(d, [([(x, 1.0) for x in e], -(len(e) - 1.0), 1.0)])
        return d
    SG = eq_pair("SG", [SI, SJ], [GI, GJ])
    SAX = eq_pair("SAX", [SI, SJ], [AX[:ND], AX[ND:]])
    GAY = eq_pair("GAY", [GI, GJ], [AY[:ND], AY[ND:]])
    XY = eq_pair("XY", [X], [Y])

    # ---- required actions (static conditions)
    REQ = net.g("REQ")[0]
    base = [(GF, 1.0), (SG, -1.0)]
    f = lambda g: [(FGr[g], 1.0)]
    net.set_relus(REQ, [
        (f(1) + base + [(XY, 1.0)], -2.0, 1.0), (f(1) + base + [(XY, -1.0), (SAX, -1.0)], -1.0, 1.0),
        (f(2) + base + [(XY, -1.0), (SAX, -1.0)], -1.0, 1.0),
        (f(3) + base + [(XY, 1.0)], -2.0, 1.0),
        (f(4) + base + [(XY, -1.0)], -1.0, 1.0),
        (f(5) + base + [(XY, -1.0), (GAY, -1.0)], -1.0, 1.0)])
    vehicle = {"tX": tX, "tY": tY}
    place = {"S": (SI, SJ), "AX": (AX[:ND], AX[ND:]), "G": (GI, GJ), "AY": (AY[:ND], AY[ND:])}
    # key slots of required moves: vehicle (truck one-hot / airplane flag) and destination
    VKt, VKp = net.g("VKt", ND), net.g("VKp")[0]
    for c in range(ND):
        net.set_relus(VKt[c], [([(KEY[v], 1.0), (vehicle[v][c], 1.0), (REQ, 1.0)], -2.0, 1.0) for v in ("tX", "tY")])
    net.set_relus(VKp, [([(KEY["pl"], 1.0), (REQ, 1.0)], -1.0, 1.0)])
    KDI, KDJ = net.g("KDI", ND), net.g("KDJ", ND)
    for c in range(ND):
        net.set_relus(KDI[c], [([(KD[d], 1.0), (place[d][0][c], 1.0)], -1.0, 1.0) for d in KD])
        net.set_relus(KDJ[c], [([(KD[d], 1.0), (place[d][1][c], 1.0)], -1.0, 1.0) for d in KD])
    # dynamic actions: query with the vehicle, find its latest earlier required move (causal, most recent)
    QVt = net.g("QVt", ND)
    for c in range(ND):
        net.set_relus(QVt[c], [([(SQ[v], 1.0), (vehicle[v][c], 1.0)], -1.0, 1.0) for v in ("tX", "tY")])
    T_cur = net.g("T_cur", 2 * ND + 1)
    net.match_head([(QVt, VKt), ([SQ["pl"]], [VKp])], None, T_cur, A=10000.0, bos=5000.0, mask="causal",
                   extra_q=[(one, 2.0)], extra_k=[(posd, 1.0)],
                   vsum=[[d] for d in KDI] + [[d] for d in KDJ] + [list(VKt) + [VKp]])
    FRI, FRJ = net.g("FRI", ND), net.g("FRJ", ND)
    found = T_cur[2 * ND]
    start = {"tX": (TXL[:ND], TXL[ND:]), "tY": (TYL[:ND], TYL[ND:]), "pl": (HI, HJ)}
    for c in range(ND):
        for dst, k in ((FRI, 0), (FRJ, 1)):
            terms = [([((T_cur[c] if k == 0 else T_cur[ND + c]), 1.0)], 0.0, 1.0)]
            terms += [([(SQ[v], 1.0), (start[v][k][c], 1.0), (found, -1.0)], -1.0, 1.0) for v in start]
            net.set_relus(dst[c], terms)
    DDI, DDJ = net.g("DDI", ND), net.g("DDJ", ND)
    for c in range(ND):
        net.set_relus(DDI[c], [([(DD[d], 1.0), (place[d][0][c], 1.0)], -1.0, 1.0) for d in DD])
        net.set_relus(DDJ[c], [([(DD[d], 1.0), (place[d][1][c], 1.0)], -1.0, 1.0) for d in DD])
    EQF = eq_pair("F", [FRI, FRJ], [DDI, DDJ])
    ACT = net.g("ACT")[0]
    net.set_relus(ACT, [([(REQ, 1.0), (DYN, -1.0)], 0.0, 1.0),
                        ([(REQ, 1.0), (DYN, 1.0), (EQF, -1.0)], -1.0, 1.0)])

    src = {"tX": tX, "tY": tY, "aK": aK, "SI": SI, "SJ": SJ, "GI": GI, "GJ": GJ, "AXI": AX[:ND],
           "AXJ": AX[ND:], "AYI": AY[:ND], "AYJ": AY[ND:], "FI": FRI, "FJ": FRJ, "X": X, "Y": Y}
    for s in fixed:
        net.set_relus(LG[s], [([(FX[s], 1.0), (ACT, 1.0)], -1.0, 1.0)])
    for dgt in range(ND):
        terms = [([(PD[dgt], 1.0), (ACT, 1.0)], -1.0, 1.0)]
        for s in SRC:
            terms.append(([(SEL[s], 1.0), (src[s][dgt], 1.0), (ACT, 1.0)], -2.0, 1.0))
        net.set_relus(LG[str(dgt)], terms)

    for gname, dims in list(net.groups.items()):
        if gname.startswith("T_"):
            net.clear(dims)
    loops = {"const": 20, "lin": 0, "div": 1, "log2": 0}
    info = net.save(out_dir, alphabet, LG, loops)
    info["canvas_positions"] = canvas_end - 1
    return info


def main():
    global HERE
    import sys
    if len(sys.argv) > 1:
        HERE = os.path.abspath(sys.argv[1])
    res = {}
    for dom in ("blocksworld", "mystery_blocksworld"):
        res[dom] = build_bw(dom, os.path.join(HERE, "net_" + dom))
    res["logistics"] = build_logistics(os.path.join(HERE, "net_logistics"))
    for k, v in res.items():
        print(k, v)


if __name__ == "__main__":
    main()
