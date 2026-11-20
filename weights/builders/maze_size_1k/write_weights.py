"""Writes every weight of the looped maze network (no training, no data).

Residual stream (d = 12):
  0 sig   parity of the token index (+1 even, -1 odd)            [embedding]
  1 P     token index - 1, built by a loop recurrence             (P_i = P_{i-1} + 1)
  2 Q     P^2, built by the recurrence Q_i = Q_{i-1} + 2 P_{i-1} + 1
  3 C,4 S cos / sin of theta * P, theta = 2 pi / 31  (column angle), built by rotation recurrence
  5 tau   loop clock 1, 2, ..., T0 then constant T0 (used as the constant and as the start gate)
  6 open  1 for ' ', 'S'            7 w   1 for '\\n' and 'S', -1 for 'G'
  8 u     0 unreached, 1 reached by the wave from G, 2 on the path (marked)
  9 pi    (index of the cell's parent toward G) + 1, set when the cell is reached
 10 T1, 11 T2  per-loop scratch written by the neighbour head (cleared by the self head next loop)

One layer, three heads:
  self  (causal)  attends to itself (latest maximal P) and subtracts P, Q, C, S, T1, T2
  left  (causal)  attends to the previous token (latest maximal P of opposite parity) and adds
                  P, Q + 2P, and the rotated (C, S): together with the self head and the MLP this
                  runs the positional recurrence; it converges front to back, one token per loop.
  nbr   (full)    score = A cos(theta dP) - dP^2 + tie-break + lam * u_j  (dP = P_j - P_x):
                  the four grid neighbours (dP = +-1, +-31) are the only cells that can beat the
                  token itself, and only if their u is larger, so each token attends to its
                  highest-priority (D > L > R > U) neighbour with larger u, or to itself.
MLP (9 units): clock saturation; first-token guard of the recurrence; BFS step (2 units);
  parent pointer store; trace step (4 units, a flat-top equality test).
"""
import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "net")

d = 12
SIG, P, Q, C, S, TAU, OPEN, W, U, PI, T1, T2 = range(12)

T0 = 945            # loops before the wave starts (the positional recurrence needs about 930)
DMAX = 230          # largest start-goal distance handled
R = T0 + 2 * DMAX + 10

th = 2 * math.pi / 31
c, s = math.cos(th), math.sin(th)

# ---------------- embeddings
E0 = np.zeros((7, d))           # ord(ch) % 7
E0[4, OPEN] = 1                 # ' '  (32)
E0[6, OPEN] = 1; E0[6, W] = 1   # 'S'  (83)
E0[3, W] = 1                    # '\n' (10)
E0[1, W] = -1; E0[1, U] = 1; E0[1, PI] = -5000   # 'G' (71): the source of the wave
# rows 0 ('#'), 2, 5 (other characters): all zero = wall

E1 = np.zeros((3, d))           # 0 first, 1 odd, 2 even
E1[:, TAU] = 1
E1[0, [SIG, P, Q, C, S]] = [1, -1, 1, c, -s]
E1[1, [SIG, P, Q, C, S]] = [-1, 0, 0, 1, 0]
E1[2, [SIG, P, Q, C, S]] = [1, 0, 0, 1, 0]

# ---------------- attention
heads = [{"dk": 1, "dv": 6, "mask": "causal"},   # self
         {"dk": 2, "dv": 4, "mask": "causal"},   # left
         {"dk": 5, "dv": 2, "mask": "none"}]     # nbr
sk, sv = 8, 12
Wq = np.zeros((d, sk)); Wk = np.zeros((d, sk)); Wv = np.zeros((d, sv)); Wo = np.zeros((sv, d))

# self head: score = gs * tau * P_j
gs = 50.0
Wq[TAU, 0] = gs * math.sqrt(1); Wk[P, 0] = 1
for k, dim in enumerate([P, Q, C, S, T1, T2]):
    Wv[dim, k] = -1; Wo[k, dim] = 1
Wv[P, 4] = -1; Wv[P, 5] = -1   # scratch also receives -P_self, so T1, T2 are relative to the token

# left head: score = gl * tau * P_j - mu * sig_i * sig_j
gl, mu = 30.0, 3.0e4
Wq[TAU, 1] = gl * math.sqrt(2); Wk[P, 1] = 1
Wq[SIG, 2] = -mu * math.sqrt(2); Wk[SIG, 2] = 1
Wv[P, 6] = 1;                      Wo[6, P] = 1
Wv[Q, 7] = 1; Wv[P, 7] = 2;        Wo[7, Q] = 1
Wv[C, 8] = c; Wv[S, 8] = -s;       Wo[8, C] = 1
Wv[C, 9] = s; Wv[S, 9] = c;        Wo[9, S] = 1

# neighbour head
A = 960 / (1 - c)          # makes the +-1 and +-31 neighbours score alike
e1 = 3.0
e2 = 10 * e1 / s           # tie-break: D (+31) > L (-1) > R (+1) > U (-31)
lam = 1400.0
B1, B2 = 1000.0, 1000.0
r5 = math.sqrt(5)
k0 = 3
Wk[C, k0 + 0] = 1; Wq[C, k0 + 0] = A * r5; Wq[S, k0 + 0] = e2 * r5
Wk[S, k0 + 1] = 1; Wq[S, k0 + 1] = A * r5; Wq[C, k0 + 1] = -e2 * r5
Wk[P, k0 + 2] = 1; Wq[P, k0 + 2] = 2 * r5; Wq[TAU, k0 + 2] = e1 / T0 * r5
Wk[Q, k0 + 3] = 1; Wq[TAU, k0 + 3] = -1 / T0 * r5
Wk[U, k0 + 4] = 1; Wq[TAU, k0 + 4] = lam / T0 * r5
Wv[P, 10] = 1; Wv[U, 10] = B1;                         Wo[10, T1] = 1
Wv[PI, 11] = 1; Wv[P, 11] = 1; Wv[U, 11] = B2; Wv[TAU, 11] = (-2 * B2 - 62) / T0; Wo[11, T2] = 1

# ---------------- MLP
F = 9
W1 = np.zeros((d, F)); b1 = np.zeros(F); W2 = np.zeros((F, d)); b2 = np.zeros(d)
# 0: clock tau -> min(tau + 1, T0)
W1[TAU, 0] = 1; b1[0] = -(T0 - 1); W2[0, TAU] = -1; b2[TAU] = 1
# 1: guard of the first token (P' = -1 and sig = +1 only there)
Kg = 10.0
W1[P, 1] = -1; W1[SIG, 1] = Kg; b1[1] = -Kg
W2[1, P] = -1; W2[1, Q] = 1; W2[1, C] = c - 1; W2[1, S] = -s
b2[P] = 1; b2[Q] = 1
# gate: open, unreached, clock at T0
K = 2.0 ** 15
def gate(j):
    W1[OPEN, j] += K; W1[U, j] += -K; W1[TAU, j] += K; b1[j] += -K - K * T0
dsc = 4 / (B1 - 31)
# 2, 3: BFS step: u += 1 (2 for 'S') when the attended token is a reached neighbour
for j in (2, 3):
    gate(j); W1[T1, j] += dsc      # T1 = P_att - P_self + B1 u_att;  d' = dsc T1 (0 self, >= 4 neighbour)
b1[2] += -1                         # e1 = relu(d' - 1 + g)
W1[W, 3] += -1; W1[OPEN, 3] += -1; b1[3] += -1   # e2 = relu(d' - 2 - zeta + g), zeta = w + open - 1 (1 only for 'S')
W2[2, U] = 1; W2[3, U] = -1
# 4: parent pointer (offset to the parent, + 62)
gate(4); W1[T1, 4] += 1; b1[4] += -B1 + 62; W2[4, PI] = 1   # pi = (P_parent - P_self) + 62
# 5..8: trace step: u += [|T2| <= 0.4], T2 = P_parent(att) - P_self  (flat-top trapezoid, exact 1 inside, 0 outside 0.6)
Kt = 2.0 ** 15
Ku = 2.0 ** 12
for j, off, wt in zip((5, 6, 7, 8), (0.6, 0.4, -0.4, -0.6), (5, -5, -5, 5)):
    W1[T2, j] = 1; W1[TAU, j] = Kt; b1[j] = off - Kt * T0
    W1[U, j] = Ku; b1[j] += -Ku    # only cells with u = 1 can be marked
    W2[j, U] = wt

# ---------------- readout
alphabet = ["", "#", "\n", " ", "o", "S", "G"]
Wout = np.zeros((d, 7)); bout = np.zeros(7)
bout[0] = -100
bout[1] = 0.5
Wout[W, 2] = 2; Wout[OPEN, 2] = -2
Wout[OPEN, 3] = 2
Wout[OPEN, 4] = 2; Wout[U, 4] = 2; Wout[W, 4] = -2; bout[4] = -3
Wout[W, 5] = 2; Wout[OPEN, 5] = 2; bout[5] = -1.5
Wout[W, 6] = -4

os.makedirs(OUT, exist_ok=True)
np.savez(os.path.join(OUT, "weights.npz"), **{
    "emb.0": E0, "emb.1": E1,
    "L0.Wq": Wq, "L0.Wk": Wk, "L0.Wv": Wv, "L0.Wo": Wo,
    "L0.W1": W1, "L0.b1": b1, "L0.W2": W2, "L0.b2": b2,
    "out.W": Wout, "out.b": bout})
json.dump({"d_model": d, "fields": ["char_mod7", "position_class"],
           "layers": [{"heads": heads, "d_ff": F}],
           "loops": {"const": R, "lin": 0, "div": 1, "log2": 0},
           "alphabet": alphabet}, open(os.path.join(OUT, "network.json"), "w"), indent=1)
print("R =", R)
