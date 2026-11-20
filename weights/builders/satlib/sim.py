"""Step-level simulator of the parallel-BCP DPLL the network implements (one step = one loop)."""
import json, sys, numpy as np

def solve(clauses, V, heur="jw", maxsteps=100000, both_lazy=True, trace=None):
    C = len(clauses)
    P = np.zeros((C, V+1), int); N = np.zeros((C, V+1), int)
    for i, cl in enumerate(clauses):
        for l in cl:
            (P if l > 0 else N)[i, abs(l)] = 1
    s = P.sum(1) + N.sum(1)
    T = np.zeros(V+1, int); F = np.zeros(V+1, int); lev = np.zeros(V+1, int); dec = np.zeros(V+1, int)
    flp = np.zeros(V+1, int); fl = np.zeros(V+2, int); L = 0
    for step in range(1, maxsteps+1):
        if trace is not None and step > 1:
            trace.append((mode, T.copy(), F.copy(), lev.copy(), L))
        X = T * (1 - F); Y = F * (1 - T)          # a variable set both ways makes both its literals false
        t = P @ X + N @ Y; f = P @ F + N @ T
        conf_c = (t == 0) & (f == s); unit = (t == 0) & (f == s-1)
        un = ((T + F) == 0).astype(int)
        fP = ((P * un[None]) [unit]).sum(0) > 0; fN = ((N * un[None])[unit]).sum(0) > 0
        conflict = conf_c.any() or ((fP & fN).any() and not both_lazy)
        if conflict:
            U = [k for k in range(1, L+1) if fl[k] == 0]
            if not U: return "UNSAT", step, None
            B = max(U)
            w = [v for v in range(1, V+1) if dec[v] and lev[v] == B][0]
            wasT = T[w]
            m = (lev >= B) & ((T + F) >= 1)
            T[m] = 0; F[m] = 0; lev[m] = 0; dec[m] = 0; flp[m] = 0
            if wasT: F[w] = 1
            else: T[w] = 1
            lev[w] = B; dec[w] = 1; flp[w] = 1
            fl[B] = 1; fl[B+1:] = 0; L = B; mode = "B"
        elif fP.any() or fN.any():
            T[fP] = 1; F[fN] = 1; lev[fP | fN] = L; mode = "P"
        elif (t > 0).all():
            a = [v if F[v] == 0 else -v for v in range(1, V+1)]
            return "SAT", step, a
        else:
            act = (t == 0)
            k = (s - f)  # unassigned literals in clause
            if heur == "jw":
                w_c = np.where(act, 2.0 ** (-k), 0)
            elif heur == "moms":
                w_c = np.where(act & (k == k[act].min()), 1.0, 0)
            sp = (w_c[:, None] * P * un[None]).sum(0); sn = (w_c[:, None] * N * un[None]).sum(0)
            sc = sp + sn; sc[0] = -1; sc[un == 0] = -1
            v = int(np.argmax(sc))
            L += 1
            if sp[v] >= sn[v]: T[v] = 1
            else: F[v] = 1
            lev[v] = L; dec[v] = 1; flp[v] = 0; fl[L] = 0; mode = "D"
    return "UNKNOWN", maxsteps, None

if __name__ == "__main__":
    heur = sys.argv[1] if len(sys.argv) > 1 else "jw"
    L = [json.loads(x) for x in open("dev/dev_satlib.jsonl")]
    res = {}
    for it in L:
        r, st, a = solve(it["clauses"], it["n_vars"], heur)
        assert r == it["label"], (it["id"], r)
        if a: assert all(any(l in set(a) for l in c) for c in it["clauses"])
        res.setdefault(it["set"], []).append(st)
    for k, v in res.items():
        v = np.array(v); print(k, "mean", v.mean(), "median", np.median(v), "max", v.max(), "p95", np.percentile(v, 95))
