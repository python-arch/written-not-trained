#!/usr/bin/env python3
"""Record what the real Maze-Hard network does, loop by loop, on one test maze (build tool, not public).

    python3 figures/make_maze_trace.py PATH/TO/loopws PATH/TO/test.csv [--index 0]

Builds the locked network with the experiment repo's own code, checks it against the locked record, runs the same
generic loop used for the test (embed, then the one shared layer again and again, then the per-cell readout), and
records, for every cell, the loop at which the wave from the goal reached it and the loop at which it joined the
path from the start. Both are read from the network's own state after each loop. Also checks that the final readout
is exactly the dataset's answer, and from which loop on nothing in the state changes.
Writes figures/maze_trace.json, which make_maze_hero.py uses.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("loopws", type=Path)
    ap.add_argument("test_csv", type=Path)
    ap.add_argument("--index", type=int, default=0)
    a = ap.parse_args()

    sys.path.insert(0, str(a.loopws / "src" / "maze"))
    import looped
    import weights as W

    P = W.build()
    assert W.sha256(P) in (a.loopws / "results" / "maze" / "frozen_maze.txt").read_text(), "weights differ from the locked record"

    rows = list(csv.reader(open(a.test_csv, newline="")))[1:]
    _, question, answer, _ = rows[a.index]
    side = int(round(len(question) ** 0.5))
    tokens = W.encode([question])
    R = W.r_min(side * side)

    h = looped.embed(P, np.asarray(tokens))
    reached = lambda h: h[0, :, W.PH:W.PH + 3].sum(-1) > 0.5
    on = lambda h: h[0, :, W.ON] > 0.5
    n = side * side
    reach_loop = np.where(reached(h), 0, -1)
    on_loop = np.where(on(h), 0, -1)
    prev_out = W.decode(looped.readout(P, h).argmax(-1))[0]
    last_change, prev_h, settled = 0, h.copy(), None
    for t in range(1, R + 1):
        h = looped.block(P, h)
        rc, oc = reached(h), on(h)
        reach_loop[(reach_loop < 0) & rc] = t
        on_loop[(on_loop < 0) & oc] = t
        out = W.decode(looped.readout(P, h).argmax(-1))[0]
        if out != prev_out:
            last_change = t
        prev_out = out
        if settled is None and np.array_equal(h, prev_h) and t > 1:
            settled = t - 1                      # the whole state is the same as after the previous loop
        elif settled is not None and not np.array_equal(h, prev_h):
            settled = None
        prev_h = h.copy()
    pointer = h[0, :, [W.PT[d] for d in "DLRU"]].T            # (n, 4)
    ptr = ["DLRU"[int(np.argmax(p))] if p.max() > 0.5 else "" for p in pointer]
    s_idx, g_idx = question.index("S"), question.index("G")
    trace = {
        "about": "Real per-loop state of the locked Maze-Hard network on one test maze.",
        "test_index": a.index, "side": side, "question": question, "answer": answer, "loops": R,
        "reach_loop": reach_loop.tolist(), "on_loop": on_loop.tolist(), "pointer": ptr,
        "start": s_idx, "goal": g_idx, "path_cells": answer.count("o"),
        "start_reached_at": int(reach_loop[s_idx]), "readout_last_change": last_change,
        "state_unchanged_from": settled, "final_readout": prev_out, "exact_match": prev_out == answer,
    }
    out = HERE / "maze_trace.json"
    out.write_text(json.dumps(trace))
    print(f"test maze {a.index}: side {side}, {R} loops, wave reaches the start at loop {trace['start_reached_at']}, "
          f"readout last changes at loop {last_change}, whole state unchanged from loop {settled}, "
          f"path cells {trace['path_cells']}, exact match {trace['exact_match']}")


if __name__ == "__main__":
    main()
