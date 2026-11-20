"""Writes every weight of the GraphWalks-BFS looped transformer.  Run: python3 -I work/write_weights.py [out_dir] [R]

Nothing is trained or fitted; every number below is set by hand-written rules.

What the network computes (one token = 4 bytes of a line, with a byte window of offsets -24..59, see tokenizer.py):

  embedding (linear in the window bytes):
    - per window offset, a 2-d circle code of the hex digit there (cos, sin of 2*pi*v/16; 0 if not a hex digit),
    - per offset -8..1, the one-hot of the hex digit there,
    - template-match counters: for chunk index c = 0..5, how many window bytes agree with an edge line
      "<10 hex> -> <10 hex>\\n" seen from chunk c; one counter for the BFS operation line seen from its chunk 6
      ("Perform a BFS from node <10 hex> and return only the nodes at exactly depth <digit>"),
    - the digit values / digit flags of the depth number, chunk-index one-hots, first/last token flags, a constant.
  block (the same every loop), three layers:
    L1  attention head "BFS": query = source-node code of the edge line the token belongs to, key = target-node
        code + gamma * reached, plus a constant sink key on all non-edge tokens; value = (reached, countdown).
        The operation token is a virtual edge "-> X", reached, countdown N + 1.
        MLP: (idempotent "set" updates) canonical source / target codes of the token's line, the hex digits this
        token would write, validity flags, the operation token's state, the depth N parsed from its digits;
        the "newly reached" gate G = [attention says a parent is reached and this line is not yet reached];
        E = |countdown|.
    L2  MLP: reached += G; countdown = parent countdown - 1 when G; clears the scratch dims;
        answer flags = reached and countdown == 0 (the source node is at exactly depth N).
  readout: per token, "" by default; on an edge line whose source is an answer node, chunks 0..4 write the
    source id two hex digits each and chunk 5 writes ","; the first token writes "Final Answer: [", the last "]".
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "net")
R_LOOPS = int(sys.argv[2]) if len(sys.argv) > 2 else 400

LO, HI = -24, 60
NWIN = HI - LO
F_CHUNK, F_FIRST, F_LAST = NWIN, NWIN + 1, NWIN + 2
HEX = "0123456789abcdef"

# ---------------------------------------------------------------- residual layout
dims = {}


def alloc(name, k=1):
    start = sum(len(v) for v in dims.values())
    dims[name] = list(range(start, start + k))
    return dims[name]


def D(name, i=0):
    return dims[name][i]


CIR_OFFS = list(range(-20, 24))      # offsets whose circle code is needed
OH_OFFS = list(range(-8, 2))         # offsets whose one-hot is needed
alloc("CONST")
for c in range(5):
    alloc(f"CHK{c}")
for o in CIR_OFFS:
    alloc(f"CIR{o}", 2)
for o in OH_OFFS:
    alloc(f"OH{o}", 16)
for c in range(6):
    alloc(f"PV{c}")                  # edge-line template counter, seen from chunk c (full = 26)
alloc("PO")                          # operation-line template counter (full = PO_FULL)
for o in (54, 55, 56):
    alloc(f"DV{o}")                  # digit value at offset o
for o in (55, 56):
    alloc(f"ID{o}")                  # is digit at offset o
alloc("FIRST"); alloc("LAST")
alloc("SRC", 20); alloc("TGT", 20)   # canonical circle codes of the line's source / target node
alloc("CH0", 16); alloc("CH1", 16)   # hex digits this token would write
for nm in ("VALIDP", "VALID5", "OPR", "OPRCD", "R", "RCD", "AS", "ARCD", "G", "E", "ANSP", "ANSC"):
    alloc(nm)
d_model = sum(len(v) for v in dims.values())

PV_FULL = 26
OP_PREFIX = "Perform a BFS from node "                           # offsets -24..-1 of the chunk-6 token
OP_MID = " and return only the nodes at exactly depth "          # offsets 10..53
assert len(OP_PREFIX) == 24 and len(OP_MID) == 44
PO_FULL = 1 + 24 + 10 + 44 + 1

# ---------------------------------------------------------------- embedding
emb = [np.zeros((257, d_model)) for _ in range(NWIN)] + [np.zeros((128, d_model)), np.zeros((2, d_model)),
                                                         np.zeros((2, d_model))]


def edge_expect(col):
    """What an edge line has at column col: 'hex', a literal byte, or None (no constraint)."""
    if 0 <= col <= 9 or 14 <= col <= 23:
        return "hex"
    if 10 <= col <= 13:
        return " -> "[col - 10]
    if col == 24:
        return "\n"
    return None


for k in range(NWIN):
    o = LO + k
    E = emb[k]
    for byte in range(256):
        row = byte + 1
        ch = chr(byte)
        ishex = ch in HEX
        if o in CIR_OFFS and ishex:
            v = HEX.index(ch)
            E[row, D(f"CIR{o}", 0)] = math.cos(2 * math.pi * v / 16)
            E[row, D(f"CIR{o}", 1)] = math.sin(2 * math.pi * v / 16)
        if o in OH_OFFS and ishex:
            E[row, D(f"OH{o}", HEX.index(ch))] = 1.0
        for c in range(6):
            ex = edge_expect(o + 4 * c)
            if ex is not None and ((ex == "hex" and ishex) or ex == ch):
                E[row, D(f"PV{c}")] += 1.0
        # operation line seen from chunk 6
        if -24 <= o <= -1 and ch == OP_PREFIX[o + 24]:
            E[row, D("PO")] += 1.0
        if 0 <= o <= 9 and ishex:
            E[row, D("PO")] += 1.0
        if 10 <= o <= 53 and ch == OP_MID[o - 10]:
            E[row, D("PO")] += 1.0
        if o == 54 and ch in "0123456789":
            E[row, D("PO")] += 1.0
        if o in (54, 55, 56) and ch in "0123456789":
            E[row, D(f"DV{o}")] = float(int(ch))
            if o != 54:
                E[row, D(f"ID{o}")] = 1.0
Ec = emb[F_CHUNK]
Ec[:, D("CONST")] = 1.0
for c in range(6):
    Ec[c, D(f"PV{c}")] += 1.0
    if c < 5:
        Ec[c, D(f"CHK{c}")] = 1.0
Ec[6, D("PO")] += 1.0
emb[F_FIRST][1, D("FIRST")] = 1.0
emb[F_LAST][1, D("LAST")] = 1.0


# ---------------------------------------------------------------- MLP builder
class MLP:
    def __init__(self):
        self.w1, self.b1, self.w2 = [], [], []

    def unit(self, inp, bias, out):
        """hidden = relu(sum inp[dim]*w + bias); residual[dim] += w * hidden for out."""
        col = np.zeros(d_model)
        for dm, w in inp:
            col[dm] += w
        oc = np.zeros(d_model)
        for dm, w in out:
            oc[dm] += w
        self.w1.append(col); self.b1.append(bias); self.w2.append(oc)

    def linear(self, inp, bias, out, signed=True):
        """residual[out] += (sum inp + bias), exactly, via relu(x) - relu(-x) (or relu(x) if x >= 0)."""
        self.unit(inp, bias, out)
        if signed:
            self.unit([(dm, -w) for dm, w in inp], -bias, [(dm, -w) for dm, w in out])

    def arrays(self):
        return (np.array(self.w1).T, np.array(self.b1), np.array(self.w2))


def gated_copy(m, src, gate_inp, gate_bias, dst):
    """dst += src * [gate == 1] for |src| <= 1, where gate = sum(gate_inp) + gate_bias is an integer <= 0 when the
    gate is open at 0 and negative otherwise: relu(x + 2g) - relu(-x + 2g)."""
    gi = [(dm, 2 * w) for dm, w in gate_inp]
    m.unit([(src, 1.0)] + gi, 2 * gate_bias, [(dst, 1.0)])
    m.unit([(src, -1.0)] + gi, 2 * gate_bias, [(dst, -1.0)])


# ---- L1 MLP
A = MLP()
# canonical source / target codes (set semantics: new - old)
for c in range(6):
    for j in range(10):
        for comp in range(2):
            gate = ([(D(f"PV{c}"), 1.0)], -PV_FULL)
            gated_copy(A, D(f"CIR{j - 4 * c}", comp), *gate, D("SRC", 2 * j + comp))
            gated_copy(A, D(f"CIR{14 + j - 4 * c}", comp), *gate, D("TGT", 2 * j + comp))
for j in range(10):
    for comp in range(2):
        gated_copy(A, D(f"CIR{j}", comp), [(D("PO"), 1.0)], -PO_FULL, D("TGT", 2 * j + comp))
for i in range(20):
    for nm in ("SRC", "TGT"):
        A.linear([(D(nm, i), -1.0)], 0.0, [(D(nm, i), 1.0)])
# hex digits to write: chunk c writes source digits 2c, 2c+1 = window offsets -2c, -2c+1
for c in range(5):
    for kk in range(2):
        for v in range(16):
            A.unit([(D(f"OH{-2 * c + kk}", v), 1.0), (D(f"CHK{c}"), 1.0)], -1.0, [(D(f"CH{kk}", v), 1.0)])
for kk in range(2):
    for v in range(16):
        A.unit([(D(f"CH{kk}", v), 1.0)], 0.0, [(D(f"CH{kk}", v), -1.0)])
# validity flags
for c in range(6):
    A.unit([(D(f"PV{c}"), 1.0)], -(PV_FULL - 1), [(D("VALIDP" if c < 5 else "VALID5"), 1.0)])
A.unit([(D("VALIDP"), 1.0)], 0.0, [(D("VALIDP"), -1.0)])
A.unit([(D("VALID5"), 1.0)], 0.0, [(D("VALID5"), -1.0)])
# operation token: reached = 1, countdown = N + 1
BIG = 1e4
po_gate = [(D("PO"), BIG)]
A.unit([(D("PO"), 1.0)], -(PO_FULL - 1), [(D("OPR"), 1.0), (D("OPRCD"), 1.0)])
A.unit([(D("DV54"), 1.0), (D("ID55"), -BIG)] + po_gate, -BIG * PO_FULL, [(D("OPRCD"), 1.0)])
A.unit([(D("DV54"), 10.0), (D("DV55"), 1.0), (D("ID55"), BIG), (D("ID56"), -BIG)] + po_gate,
       -BIG - BIG * PO_FULL, [(D("OPRCD"), 1.0)])
A.unit([(D("DV54"), 100.0), (D("DV55"), 10.0), (D("DV56"), 1.0), (D("ID56"), BIG)] + po_gate,
       -BIG - BIG * PO_FULL, [(D("OPRCD"), 1.0)])
A.unit([(D("OPR"), 1.0)], 0.0, [(D("OPR"), -1.0)])
A.unit([(D("OPRCD"), 1.0)], 0.0, [(D("OPRCD"), -1.0)])
# newly-reached gate G = clamp(z, 0, 1), z = K (AS - 0.5) - K R - K (1 - valid)
K = 10.0
zin = [(D("AS"), K), (D("R"), -K), (D("VALIDP"), K), (D("VALID5"), K)]
A.unit(zin, -0.5 * K - K, [(D("G"), 1.0)])
A.unit(zin, -0.5 * K - K - 1.0, [(D("G"), -1.0)])
# E = |RCD| (set)
A.unit([(D("RCD"), 1.0)], 0.0, [(D("E"), 1.0)])
A.unit([(D("RCD"), -1.0)], 0.0, [(D("E"), 1.0)])
A.unit([(D("E"), 1.0)], 0.0, [(D("E"), -1.0)])

# ---- L2 MLP
B = MLP()
B.unit([(D("G"), 1.0)], 0.0, [(D("R"), 1.0), (D("G"), -1.0)])
M = 1e4
B.unit([(D("ARCD"), 1.0), (D("G"), M)], -1.0 - M, [(D("RCD"), 1.0)])
B.unit([(D("ARCD"), -1.0), (D("G"), M)], 1.0 - M, [(D("RCD"), -1.0)])
B.unit([(D("AS"), 1.0)], 0.0, [(D("AS"), -1.0)])
B.linear([(D("ARCD"), -1.0)], 0.0, [(D("ARCD"), 1.0)])
for nm, vd in (("ANSP", "VALIDP"), ("ANSC", "VALID5")):
    # ans = clamp(z, 0, 1), z = 10 (0.45 - E) - 10 (1 - R) - 10 (1 - valid): exactly 1 for E <= 0.35
    zz = [(D("E"), -10.0), (D("R"), 10.0), (D(vd), 10.0)]
    B.unit(zz, 4.5 - 20.0, [(D(nm), 1.0)])
    B.unit(zz, 4.5 - 20.0 - 1.0, [(D(nm), -1.0)])
    B.unit([(D(nm), 1.0)], 0.0, [(D(nm), -1.0)])

# ---------------------------------------------------------------- BFS head
BETA, GAMMA, DELTA = 600.0, 25.0, 12.0
DK, DV = 21, 2
sq = math.sqrt(DK)
Wq = np.zeros((d_model, DK)); Wk = np.zeros((d_model, DK)); Wv = np.zeros((d_model, DV)); Wo = np.zeros((DV, d_model))
for i in range(20):
    Wq[D("SRC", i), i] = BETA * sq
    Wk[D("TGT", i), i] = 1.0
Wq[D("CONST"), 20] = sq
SINK = 10 * BETA - DELTA
Wk[D("R"), 20] = GAMMA
Wk[D("OPR"), 20] = GAMMA - SINK
Wk[D("CONST"), 20] = SINK
Wk[D("VALIDP"), 20] = -SINK
Wk[D("VALID5"), 20] = -SINK
Wv[D("R"), 0] = 1.0; Wv[D("OPR"), 0] = 1.0
Wv[D("RCD"), 1] = 1.0; Wv[D("OPRCD"), 1] = 1.0
Wo[0, D("AS")] = 1.0; Wo[1, D("ARCD")] = 1.0

# ---------------------------------------------------------------- readout
alphabet = ["", "Final Answer: [", "]", ","] + [a + b for a in HEX for b in HEX]
Wout = np.zeros((d_model, len(alphabet))); bout = np.zeros(len(alphabet))
bout[0] = 3.5
Wout[D("FIRST"), 1] = 4.0
Wout[D("LAST"), 2] = 4.0
Wout[D("ANSC"), 3] = 4.0
for ia, a in enumerate(HEX):
    for ib, b in enumerate(HEX):
        col = 4 + 16 * ia + ib
        Wout[D("CH0", ia), col] = 1.0
        Wout[D("CH1", ib), col] = 1.0
        Wout[D("ANSP"), col] = 2.0

# ---------------------------------------------------------------- write
os.makedirs(OUT, exist_ok=True)
W = {}


def put(name, a):
    a = np.asarray(a, dtype=np.float64)
    nz = np.flatnonzero(a)
    if nz.size < a.size // 4:
        W[name + ".shape"] = np.array(a.shape, dtype=np.int64)
        W[name + ".idx"] = nz.astype(np.int64)
        W[name + ".val"] = a.ravel()[nz]
    else:
        W[name] = a


for f, E in enumerate(emb):
    put(f"emb.{f}", E)
zb = np.zeros(d_model)
w1, b1, w2 = A.arrays()
put("L0.Wq", Wq); put("L0.Wk", Wk); put("L0.Wv", Wv); put("L0.Wo", Wo)
put("L0.W1", w1); put("L0.b1", b1); put("L0.W2", w2); put("L0.b2", zb)
w1b, b1b, w2b = B.arrays()
put("L1.W1", w1b); put("L1.b1", b1b); put("L1.W2", w2b); put("L1.b2", zb)
put("out.W", Wout); put("out.b", bout)
np.savez(os.path.join(OUT, "weights.npz"), **W)

cfg = {"d_model": d_model,
       "fields": [f"byte@{o:+d}" for o in range(LO, HI)] + ["chunk_index", "is_first", "is_last"],
       "layers": [{"heads": [{"dk": DK, "dv": DV, "mask": "none"}], "d_ff": int(w1.shape[1])},
                  {"heads": [], "d_ff": int(w1b.shape[1])}],
       "loops": {"const": R_LOOPS, "lin": 0, "div": 1, "log2": 0},
       "alphabet": alphabet}
json.dump(cfg, open(os.path.join(OUT, "network.json"), "w"), indent=1)
if os.path.abspath(OUT) != os.path.join(HERE, "net"):
    import shutil
    shutil.copy(os.path.join(HERE, "net", "tokenizer.py"), os.path.join(OUT, "tokenizer.py"))
tot = sum(int(np.prod(W[k + ".shape"])) if k + ".shape" in W else W[k].size
          for k in {n.rsplit(".", 1)[0] if n.endswith((".shape", ".idx", ".val")) else n for n in W})
nnz = sum(int(np.count_nonzero(v)) for k, v in W.items() if not k.endswith((".shape", ".idx")))
print(f"d_model {d_model}  d_ff {w1.shape[1]}, {w1b.shape[1]}  R {R_LOOPS}  params {tot}  nonzero {nnz}")
