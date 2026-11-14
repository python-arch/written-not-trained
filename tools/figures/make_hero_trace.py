#!/usr/bin/env python3
"""Record what the real Tower of Hanoi network outputs after each pass, for the first moves of the 20-disk game.

    python3 figures/make_hero_trace.py PATH/TO/loopws [--moves 15]

Runs the locked network (its code and weights live in the experiment repo) with the same generic loop used for the
test, reads its output after passes 1 to 4 at every step, and records its real attention over earlier tokens.
Writes figures/hero_trace.json, which make_hero.py uses for the move-by-move part of the animation.
"""
from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("loopws", type=Path)
    ap.add_argument("--moves", type=int, default=15)
    ap.add_argument("--passes", type=int, default=4)
    a = ap.parse_args()

    sys.path.insert(0, str(a.loopws / "src" / "puzzles"))
    import looped
    import weights_hanoi as W

    P = W.build()
    locked = (a.loopws / "results" / "puzzles" / "frozen_hanoi.txt").read_text()
    assert W.sha256(P) in locked, "weights differ from the locked record"

    n_disks, R = 20, a.passes
    passes = list(range(1, R + 1))
    prompt = W.prompt_ids(n_disks)
    dec = looped.Decoder(P, R, max_len=len(prompt) + a.moves + 4)
    logits = None
    for t in prompt:
        logits = dec.step(int(t), passes)

    def name(i):
        if i == W.END:
            return {"end": True}
        if i < W.N_MOVES:
            d, (f, t) = i // 6 + 1, W.PAIRS[i % 6]
            return {"disk": d, "from": f, "to": t}
        return {"other": int(i)}

    def attention_of_last(seq):
        """Real attention weights of the last token over all tokens so far, at every pass (head by head)."""
        out = {}
        hs = looped.run_full(P, seq, R, read_at=passes, return_h=True)
        for r in passes:
            h_in = P["embed"][np.asarray(seq)] if r == 1 else hs[r - 1][1]
            layer_weights = []
            for L in P["layers"]:
                if not L["H"]:
                    continue
                H = L["H"]
                dqk = L["Wq"].shape[1] // H
                q = (h_in[-1] @ L["Wq"]).reshape(H, dqk)
                k = (h_in @ L["Wk"]).reshape(len(seq), H, dqk).transpose(1, 0, 2)
                s = (k @ q[:, :, None])[:, :, 0] / np.sqrt(dqk)
                s = s - s.max(axis=1, keepdims=True)
                w = np.exp(s) / np.exp(s).sum(axis=1, keepdims=True)
                layer_weights.append(w)
            out[r] = layer_weights
        return out

    seq, steps = list(prompt), []
    for k in range(a.moves):
        args = {r: int(np.argmax(logits[r])) for r in passes}
        same_after = [r for r in passes if r > W.R_MIN and np.array_equal(logits[r], logits[W.R_MIN])]
        att = attention_of_last(seq)
        n = len(seq)
        dev = max(float(np.abs(w - 1.0 / n).max()) for r in passes for w in att[r])
        tok = args[W.R_MIN]                       # the token the network writes (same rule as the test)
        steps.append({
            "move": k + 1,
            "tokens_so_far": n,
            "output_by_pass": {str(r): name(args[r]) for r in passes},
            "written": name(tok),
            "identical_to_pass_2": [str(r) for r in same_after],
            "attention_uniform_max_deviation": dev,
        })
        seq.append(tok)
        logits = dec.step(tok, passes)

    saved = gzip.open(a.loopws / "results/puzzles/answers/hanoi_test_torch_N20.txt.gz", "rt").read()
    saved_moves = [tuple(map(int, m)) for m in re.findall(r"\[(\d+), (\d+), (\d+)\]", saved)[: a.moves]]
    ours = [(s["written"]["disk"], s["written"]["from"], s["written"]["to"]) for s in steps]
    trace = {
        "about": "Real outputs of the locked Tower of Hanoi network after each pass, 20-disk game, first moves.",
        "disks": n_disks, "passes_read": passes, "pass_needed": W.R_MIN,
        "matches_saved_test_answer": ours == saved_moves,
        "steps": steps,
    }
    out = HERE / "hero_trace.json"
    out.write_text(json.dumps(trace, indent=1))
    print(f"wrote {out.name}: {len(steps)} moves, matches saved test answer: {trace['matches_saved_test_answer']}")
    for s in steps[:4]:
        print(s["move"], s["output_by_pass"], "identical:", s["identical_to_pass_2"],
              "attention dev:", s["attention_uniform_max_deviation"])


if __name__ == "__main__":
    main()
