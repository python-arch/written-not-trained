#!/usr/bin/env python3
"""Run the hand-written looped transformers in networks/ (weights set by formulas, no training). numpy, float64.

Every network runs the same generic loop; only its config (networks/<name>.json) and weights (<name>.npz) differ:
  h = sum over the token's fields f of E_f[value_f]                  (embedding: one table per field)
  repeat R times, for each layer of the ONE shared block:
      h += concat_heads(softmax(q k^T / sqrt(d_k) + mask) v) W_o        (softmax attention, no norm)
      h += relu(h W1 + b1) W2 + b2                                     (ReLU MLP)
  logits = h W_out + b_out                                           (linear readout: last token, or every token)
  greedy decoding: append the argmax id until the end id, max_new ids, or a full context window.
Masks: "none" (bidirectional; the whole sequence is rerun from scratch at every step), "causal" (every token sees
itself and earlier tokens), "prefix" (input tokens see the whole input; each generated token sees the input, earlier
generated tokens and itself). Under "causal" and "prefix" earlier rows never see later ones, so keys and values are
cached (exact). R comes from the config's rule. Only prompt -> ids, ids -> answer and the answer checks are per task.

  python run.py hanoi 10             python run.py boolean_expressions "( ( False ) ) or ( True ) is"
  python run.py checker 5            python run.py web_of_lies question.txt   (a file, or the question text)
  python run.py parity 0110101       python run.py chains "x3 = 7; x5 = x3 + 4; x9 = x5 - 2; ? x9"
  python run.py graph "0>1 1>2 2>3 5>0 ? 0 3"      python run.py graph --random 30   (also parity, chains, maze)
  python run.py maze maze.txt        (rows of "#", " ", "S", "G"; any side up to 128)
"""
import ast
import json
import math
import os
import random
import re
import sys

import numpy as np

NETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "networks")


def softmax(s):  # keep exactly this form (see README): scores reach ~1e13
    s = s - s.max(axis=-1, keepdims=True)
    e = np.exp(s)
    return e / e.sum(axis=-1, keepdims=True)


class Sparse:
    """A weight matrix kept as its nonzero entries: x @ W = per-column sums of x[:, row] * value."""
    __array_ufunc__ = None  # makes `ndarray @ Sparse` call __rmatmul__

    def __init__(self, shape, idx, val):
        r, c = np.divmod(idx.astype(np.int64), shape[1])
        o = np.lexsort((r, c))
        self.ncol, self.rows, self.vals = shape[1], r[o], val[o]
        self.cols, self.starts = np.unique(c[o], return_index=True)

    def __rmatmul__(self, x):
        out = np.zeros((x.shape[0], self.ncol))
        if len(self.vals):
            out[:, self.cols] = np.add.reduceat(x[:, self.rows] * self.vals, self.starts, axis=1)
        return out


class Cache:
    """Keys and values of the earlier rows for one (loop, layer)."""

    def __init__(self):
        self.n, self.K, self.V = 0, None, None

    def add(self, K, V):
        n = self.n + len(K)
        if self.K is None or n > len(self.K):
            K2, V2 = np.empty((2 * n, K.shape[1])), np.empty((2 * n, V.shape[1]))
            if self.n:
                K2[:self.n], V2[:self.n] = self.K[:self.n], self.V[:self.n]
            self.K, self.V = K2, V2
        self.K[self.n:n], self.V[self.n:n] = K, V
        self.n = n
        return self.K[:n], self.V[:n]


def load(name):
    cfg = json.load(open(os.path.join(NETS, name + ".json")))
    z = np.load(os.path.join(NETS, name + ".npz"))
    shape = lambda k: tuple(int(s) for s in z[k + ".shape"])

    def dense(k):
        a = np.zeros(math.prod(shape(k)))
        a[z[k + ".idx"].astype(np.int64)] = z[k + ".val"]
        return a.reshape(shape(k))
    sparse = lambda k: Sparse(shape(k), z[k + ".idx"], z[k + ".val"])
    net = {"cfg": cfg, "emb": [dense("emb." + f) for f in cfg["fields"]], "W_out": sparse("out.W"),
           "b_out": dense("out.b"), "layers": []}
    for l, heads in enumerate(cfg["layers"]):
        L = {"heads": heads, "W1": sparse(f"L{l}.W1"), "b1": dense(f"L{l}.b1"), "W2": sparse(f"L{l}.W2"),
             "b2": dense(f"L{l}.b2")}
        if heads:
            L.update({k: sparse(f"L{l}.{k}") for k in ("Wq", "Wk", "Wv", "Wo")})
        net["layers"].append(L)
    return net


def layer(L, h, cache):
    """One layer for the new rows h. They see the cached rows (if any) and each other (one row, or a whole input)."""
    if L["heads"]:
        Q, K, V = h @ L["Wq"], h @ L["Wk"], h @ L["Wv"]
        if cache is not None:
            K, V = cache.add(K, V)
        outs, q0, v0 = [], 0, 0
        for dk, dv in L["heads"]:
            s = Q[:, q0:q0 + dk] @ K[:, q0:q0 + dk].T / np.sqrt(dk)
            outs.append(softmax(s) @ V[:, v0:v0 + dv])
            q0, v0 = q0 + dk, v0 + dv
        h = h + np.hstack(outs) @ L["Wo"]
    return h + np.maximum(h @ L["W1"] + L["b1"], 0.0) @ L["W2"] + L["b2"]


def forward(net, toks, p0, R, caches=None):
    """Tokens (lists of field values) at positions p0, p0+1, ... -> logits of the readout rows."""
    T = np.asarray(toks, dtype=np.int64)
    if net["cfg"]["position_field"]:
        T = np.column_stack([T, p0 + np.arange(len(T))])
    h = net["emb"][0][T[:, 0]].copy()
    for f in range(1, T.shape[1]):
        h += net["emb"][f][T[:, f]]
    for t in range(R):
        for l, L in enumerate(net["layers"]):
            h = layer(L, h, caches[t][l] if caches else None)
    rows = h if net["cfg"]["readout"] == "all" else h[-1:]
    return rows @ net["W_out"] + net["b_out"]


def loops(rule, n_input, n_call):
    """R from the config rule; n = the input length or the length of this forward call."""
    n = n_call if rule["n"] == "call" else n_input
    return rule["const"] + rule["lin"] * -(-n // rule["div"]) + rule["log2"] * math.ceil(math.log2(max(n, 2)))


def generate(net, toks):
    """Generic greedy decoding. Returns the generated ids (end id excluded); readout "all": one id per position."""
    c = net["cfg"]
    if len(toks) > c["max_tokens"]:
        raise ValueError("input longer than the context window")
    R = loops(c["loops"], len(toks), len(toks))
    if c["readout"] == "all":
        return [int(i) for i in forward(net, toks, 0, R).argmax(-1)]
    seq, gen = list(toks), []
    caches = [[Cache() for _ in net["layers"]] for _ in range(R)] if c["mask"] != "none" else None
    if c["mask"] == "causal":
        for i, tk in enumerate(toks):
            lg = forward(net, [tk], i, R, caches)
    else:
        lg = forward(net, toks, 0, R, caches)
    while True:
        o = int(np.argmax(lg[-1]))
        if o == c["end_id"]:
            break
        gen.append(o)
        if len(gen) == c["max_new"] or len(seq) >= c["max_tokens"]:
            break
        tok = list(c["out_token"]["template"])
        tok[c["out_token"]["slot"]] = o
        seq.append(tok)
        if caches:
            lg = forward(net, [tok], len(seq) - 1, R, caches)
        else:
            lg = forward(net, seq, 0, loops(c["loops"], len(toks), len(seq)))
    return gen


# ------------------------------------------------------------------ task-specific I/O: prompt -> ids, ids -> answer
def prompt(name, io, arg):
    if name in ("hanoi", "checker"):
        n, ix = int(arg), {s: i for i, s in enumerate(io["vocab"]) if s}
        if name == "hanoi":
            words = [f"INIT({d},0)" for d in range(n, 0, -1)] + [f"GOAL({d},2)" for d in range(n, 0, -1)]
        else:
            init, goal = "R" * n + "_" + "B" * n, "B" * n + "_" + "R" * n
            words = [f"INIT({x},{s})" for x, s in enumerate(init)] + [f"GOAL({x},{s})" for x, s in enumerate(goal)]
        return [[ix[w]] for w in [f"N={n}"] + words + ["SEP"]]
    if name == "maze":  # rows of "# SG" (or one line of side*side); one token (row, column, character) per cell
        n = math.isqrt(len(arg)) if "\n" not in arg else len(arg.split("\n"))
        rows = arg.split("\n") if "\n" in arg else [arg[i:i + n] for i in range(0, n * n, n)]
        return [[r, c, io["charset"].index(ch)] for r, row in enumerate(rows) for c, ch in enumerate(row.ljust(n))]
    if name == "parity":  # BOS, then one token per bit
        return [[io["bos"]]] + [[int(b)] for b in arg.strip()]
    if name == "chains":  # "x5 = x3 + 4" -> (5, 3, 4); "x5 = x3 - 4" -> (5, 3, 104); "x3 = 7" -> (3, LIT, 7); "? x9"
        *st, q = [w.split() for w in arg.replace("\n", ";").split(";") if w.strip()]
        v = lambda w: int(w[1:])
        return ([[v(w[0]), io["lit"], int(w[2])] if len(w) == 3 else [v(w[0]), v(w[2]), 100 * (w[3] == "-") + int(w[4])]
                 for w in st] + [[io["query"], v(q[1]), 0]])
    if name == "graph":  # "u>v" an edge, "v" a node, in token order; nodes never listed get a token first; "? A B"
        body, q = arg.split("?")
        k, words, (a, b) = io["kind"], body.split(), map(int, q.split())
        listed = {int(w) for w in words if ">" not in w}
        ids = sorted({int(x) for w in words for x in w.split(">")} | {a, b})
        toks = [[k["node"], i, i] for i in ids if i not in listed]
        toks += [[k["edge"]] + [int(x) for x in w.split(">")] if ">" in w else [k["node"], int(w), int(w)] for w in words]
        return toks + [[k["query"], a, b]]
    if "vocab" in io:  # BBH: one token per character, whitespace runs -> one space
        ix = {s: i for i, s in enumerate(io["vocab"]) if len(s) == 1}
        text = " ".join(arg.split())
        return [[io["bos"]]] + [[ix.get(ch, io["unk"])] for ch in text] + [[io["sep"]]]
    L, k = io["piece_len"], io["kind"]  # BBEH: word pieces, each with its character codes in L slots
    code = lambda ch: io["codes"].index(ch) + 1 if ch in io["codes"] else (
        io["code_newline"] if ch == "\n" else io["code_other"])
    pieces = [m[i:i + L] for m in re.findall(io["piece_regex"], arg) for i in range(0, len(m), L)]
    special = lambda kind: [kind] + [0] * (L + 1)
    return ([special(k["bos"])] + [[k["piece"]] + [code(ch) for ch in p] + [0] * (L - len(p) + 1) for p in pieces]
            + [special(k["sep"])])


def answer(name, io, gen):
    if name in ("hanoi", "checker"):
        return "moves = [" + ", ".join(io["vocab"][i] for i in gen) + "]"
    if name == "maze":
        n = math.isqrt(len(gen))
        return "\n".join("".join(io["charset"][i] for i in gen[r * n:(r + 1) * n]) for r in range(n))
    if name in ("parity", "chains", "graph"):
        return "unreachable" if gen[0] == io.get("unreachable") else str(gen[0])
    voc = io.get("vocab", io.get("out_vocab"))
    return "".join(voc[i] if len(voc[i]) == 1 else "" for i in gen)


def check(name, n, io, gen):
    """Small independent simulator: (every move legal, goal reached, number of moves, optimal number)."""
    try:
        moves = [ast.literal_eval(io["vocab"][i]) for i in gen]
    except (ValueError, SyntaxError):
        moves = None
    if name == "hanoi":
        best, state, goal = 2 ** n - 1, [list(range(n, 0, -1)), [], []], [[], [], list(range(n, 0, -1))]
        legal = lambda d, f, t: f != t and state[f][-1:] == [d] and (state[t] == [] or state[t][-1] > d)
    else:
        best, state, goal = (n + 1) ** 2 - 1, list("R" * n + "_" + "B" * n), list("B" * n + "_" + "R" * n)
        legal = lambda c, f, t: (0 <= min(f, t) and max(f, t) < len(state) and state[f] == c and state[t] == "_"
                                 and (t - f if c == "R" else f - t) in (1, 2)
                                 and (abs(t - f) == 1 or state[(f + t) // 2] == {"R": "B", "B": "R"}[c]))
    valid = moves is not None
    for mv in moves or []:
        if not legal(*mv):
            valid = False
            break
        if name == "hanoi":
            state[mv[2]].append(state[mv[1]].pop())
        else:
            state[mv[1]], state[mv[2]] = "_", mv[0]
    return valid, valid and state == goal, len(gen), best


def truth(name, toks, io):
    """Independent answer from the prompt: count the ones, follow the links, or BFS."""
    if name == "parity":
        return str(sum(t[0] for t in toks[1:]) % 2)
    if name == "chains":
        defs = {t[0]: t[1:] for t in toks[:-1]}
        val = lambda x: defs[x][1] if defs[x][0] == io["lit"] else val(defs[x][0]) + (
            defs[x][1] if defs[x][1] < 100 else 100 - defs[x][1])
        return str(val(toks[-1][1]))
    a, b = toks[-1][1:]
    edges, dist, todo = [t[1:] for t in toks if t[0] == io["kind"]["edge"]], {a: 0}, [a]
    for u in todo:
        for x, y in edges:
            if x == u and y not in dist:
                dist[y] = dist[u] + 1
                todo.append(y)
    return str(dist[b]) if b in dist else "unreachable"


def maze_check(q, out, n):
    """q, out: grids as n*n characters. (valid: the 'o' cells plus S and G form one simple S -> G path over open cells;
    shortest; equal to the shortest path that is smallest in move order D < L < R < U, the dataset's tie-break)."""
    nb = lambda p: [(p[0] + a, p[1] + b) for a, b in ((1, 0), (0, -1), (0, 1), (-1, 0))]          # D, L, R, U
    s, g = (divmod(q.index(ch), n) for ch in "SG")
    dist, todo = {g: 0}, [g]                                        # BFS from G
    for p in todo:
        for x in nb(p):
            if 0 <= min(x) and max(x) < n and q[x[0] * n + x[1]] != "#" and x not in dist:
                dist[x] = dist[p] + 1
                todo.append(x)
    best, p = list(q), s                                            # from S, always the first neighbour one step closer
    while s in dist and p != g:
        p = next(x for x in nb(p) if dist.get(x) == dist[p] - 1)
        best[p[0] * n + p[1]] = "o" if p != g else "G"
    marked = {divmod(i, n) for i, ch in enumerate(out) if ch in "oSG"}
    valid = len(out) == len(q) and all(x == y or (x, y) == (" ", "o") for x, y in zip(q, out)) and all(
        sum(x in marked for x in nb(p)) == (1 if p in (s, g) else 2) for p in marked)
    seen, todo = {s}, [s]
    for p in todo:
        for x in nb(p):
            if x in marked and x not in seen:
                seen.add(x)
                todo.append(x)
    valid = valid and seen == marked
    return valid, valid and len(marked) - 1 == dist.get(s), "".join(best) == out


def random_input(name, n):
    """parity: n random bits; chains: a chain of n hops (values kept in 0..99), shuffled; graph: n nodes, ~1.5 n edges."""
    rng = random.Random()
    if name == "maze":  # a random spanning tree over the odd cells, n extra openings, S and G on two open cells
        g, todo = [["#"] * n for _ in range(n)], [(1, 1)]
        g[1][1] = " "
        while todo:
            r, c = todo[-1]
            nxt = [(r + a, c + b) for a, b in ((2, 0), (-2, 0), (0, 2), (0, -2))
                   if 0 < r + a < n - 1 and 0 < c + b < n - 1 and g[r + a][c + b] == "#"]
            if not nxt:
                todo.pop()
                continue
            r2, c2 = rng.choice(nxt)
            g[(r + r2) // 2][(c + c2) // 2] = g[r2][c2] = " "
            todo.append((r2, c2))
        for _ in range(n):
            g[rng.randrange(1, n - 1)][rng.randrange(1, n - 1)] = " "
        (a, b), (c, d) = rng.sample([(r, c) for r in range(n) for c in range(n) if g[r][c] == " "], 2)
        g[a][b], g[c][d] = "S", "G"
        return "\n".join("".join(row) for row in g)
    if name == "parity":
        return "".join(rng.choice("01") for _ in range(n))
    ids = rng.sample(range(64), 64)
    if name == "chains":
        x, st = rng.randint(0, 99), [f"x{ids[0]} = "]
        st[0] += str(x)
        for i in range(1, n + 1):
            d = rng.choice((1, -1)) * rng.randint(1, 9)
            d = d if 0 <= x + d <= 99 else -d
            x += d
            st.append(f"x{ids[i]} = x{ids[i - 1]} {'+-'[d < 0]} {abs(d)}")
        rng.shuffle(st)
        return "; ".join(st) + f"; ? x{ids[n]}"
    edges = set()
    while len(edges) < min(round(1.5 * n), n * (n - 1)):
        edges.add(tuple(rng.sample(ids[:n], 2)))
    return " ".join(f"{u}>{v}" for u, v in edges) + " ? %d %d" % tuple(rng.sample(ids[:n], 2))


def main():
    if len(sys.argv) == 4 and sys.argv[2] == "--random":
        name, arg = sys.argv[1], random_input(sys.argv[1], int(sys.argv[3]))
        print("input:" + ("\n" if "\n" in arg else " ") + arg)
    elif len(sys.argv) == 3:
        name, arg = sys.argv[1], sys.argv[2]
    else:
        sys.exit(__doc__)
    if os.path.isfile(arg):
        arg = open(arg).read()
        arg = arg[:-1] if arg.endswith("\n") else arg     # an editor's final newline is not part of the question
    net = load(name)
    io = net["cfg"]["io"]
    toks = prompt(name, io, arg)
    gen = generate(net, toks)
    ans = answer(name, io, gen)
    print(ans)
    if name in ("hanoi", "checker"):
        valid, goal, m, best = check(name, int(arg), io, gen)
        print(f"check: valid {valid}, goal reached {goal}, moves {m} (optimal {best}: "
              f"{'2^N - 1' if name == 'hanoi' else '(N+1)^2 - 1'})")
    if name == "maze":
        q = "".join(io["charset"][t[2]] for t in toks)
        valid, shortest, smallest = maze_check(q, ans.replace("\n", ""), math.isqrt(len(toks)))
        print(f"check: valid path {valid}, shortest {shortest}, the D<L<R<U-smallest shortest path (dataset tie-break) "
              f"{smallest}")
    if name in ("parity", "chains", "graph"):
        how = {"parity": "counting the ones", "chains": "following the links", "graph": "BFS"}[name]
        t = truth(name, toks, io)
        print(f"check: {how} gives {t}; answer correct: {ans == t}")


if __name__ == "__main__":
    main()
