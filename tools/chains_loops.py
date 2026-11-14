"""Accuracy of the hand-written chains network at many loop counts, on chains of 1-8 steps.

Usage: python3 -I chains_loops.py <weights dir> <out json>
Loop counts are multiples of the input length in tokens, the same rule the network and the trained
looped baseline both use by default (1x). Each item mixes in extra chains that lead nowhere.
"""
import importlib.util, json, random, sys, time
import numpy as np

wdir, out = sys.argv[1], sys.argv[2]
spec = importlib.util.spec_from_file_location("runmod", f"{wdir}/run.py")
run = importlib.util.module_from_spec(spec)
sys.argv = [sys.argv[0]]
spec.loader.exec_module(run)

net = run.load("chains")
io = net["cfg"]["io"]
rng = random.Random(7)


def item(k):
    """One chain of k steps to the queried variable, plus 1-2 distractor chains of 1-4 steps."""
    ids = rng.sample(range(64), 64)
    pos, st, x = 0, [], rng.randint(0, 99)

    def chain(length, start_val):
        nonlocal pos
        v, names = start_val, [ids[pos]]
        pos += 1
        lines = [f"x{names[0]} = {v}"]
        for _ in range(length):
            d = rng.choice((1, -1)) * rng.randint(1, 9)
            d = d if 0 <= v + d <= 99 else -d
            v += d
            names.append(ids[pos]); pos += 1
            lines.append(f"x{names[-1]} = x{names[-2]} {'+-'[d < 0]} {abs(d)}")
        return lines, names[-1], v

    main, q, ans = chain(k, x)
    st += main
    for _ in range(rng.randint(1, 2)):
        extra, _, _ = chain(rng.randint(1, 4), rng.randint(0, 99))
        st += extra
    rng.shuffle(st)
    return "; ".join(st) + f"; ? x{q}", ans


MULTS = [1, 2, 4, 8, 16, 32, 64, 128]
N_PER_K = 40
items = [(k, *item(k)) for k in range(1, 9) for _ in range(N_PER_K)]
res = {m: [] for m in MULTS}
loops_used = {m: [] for m in MULTS}
t0 = time.time()
for k, q, ans in items:
    toks = run.prompt("chains", io, q)
    n = len(toks)
    for m in MULTS:
        R = m * n
        lg = run.forward(net, toks, 0, R)
        o = int(np.argmax(lg[-1]))
        got = run.answer("chains", io, [o])
        res[m].append(str(got).strip() == str(ans))
        loops_used[m].append(R)
summary = {
    "task": "reasoning chains, 1-8 steps, with 1-2 distractor chains per item",
    "items": len(items),
    "per_length": N_PER_K,
    "multiples": MULTS,
    "accuracy": [round(100 * sum(res[m]) / len(res[m]), 1) for m in MULTS],
    "mean_loops": [round(sum(loops_used[m]) / len(loops_used[m]), 1) for m in MULTS],
    "max_loops": [max(loops_used[m]) for m in MULTS],
    "seconds": round(time.time() - t0, 1),
}
json.dump(summary, open(out, "w"), indent=1)
print(json.dumps(summary))
