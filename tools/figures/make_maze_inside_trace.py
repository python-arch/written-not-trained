#!/usr/bin/env python3
"""Record the real internal state of the public Maze-Hard network after every loop, on test maze 1 (build tool).

    python3 figures/make_maze_inside_trace.py [--weights PATH/TO/public_weights]

Uses only the public package: weights/run.py and weights/networks/maze.npz. It embeds the maze with run.prompt, then
applies run.layer (the one shared layer) R times, exactly as run.forward does, and reads the residual stream after every
loop. Residual dims (src/maze/EXPLAIN.md): 7-9 distance from the goal mod 3 (one-hot), 10-13 next step (D, L, R, U),
14 on the path, 15-34 scratch.

Checks, all asserted:
  - the per-cell events below rebuild the stored distance, next step and path flag exactly, at every loop
    (so each one is set once and never changes);
  - every non-positional value is exactly 0 or 1 after every loop, and the scratch is exactly 0;
  - the final grid equals the dataset's answer and run.generate's output;
  - from some loop on, the whole state is bit-identical to the final state.
Writes figures/maze_inside_trace.json, which make_maze_inside.py uses.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PH, PTR, ON, SCRATCH = 7, 10, 14, 15
DIRS = "DLRU"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, default=HERE.parent / "weights")
    a = ap.parse_args()
    spec = importlib.util.spec_from_file_location("run", a.weights / "run.py")
    run = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run)

    old = json.loads((HERE / "maze_trace.json").read_text())        # test maze 1 (test.csv row 0) and its answer
    question, answer = old["question"], old["answer"]
    js = (HERE.parent / "demo" / "maze_tests.js").read_text()
    assert question in js and answer in js, "test maze 1 differs from the demo's copy of the test set"

    net = run.load("maze")
    io, cfg = net["cfg"]["io"], net["cfg"]
    side = int(round(len(question) ** 0.5))
    n = side * side
    toks = run.prompt("maze", io, question)
    R = run.loops(cfg["loops"], len(toks), len(toks))
    S, G = question.index("S"), question.index("G")

    T = np.asarray(toks, dtype=np.int64)
    h = net["emb"][0][T[:, 0]].copy()
    for f in range(1, T.shape[1]):
        h += net["emb"][f][T[:, f]]
    assert h[G, PH] == 1 and h[S, ON] == 1, "residual layout differs from EXPLAIN.md"

    def read(h):
        ph = h[:, PH:PH + 3]
        ptr = h[:, PTR:PTR + 4]
        return ph.argmax(1) * (ph.sum(1) > 0.5) - (ph.sum(1) < 0.5), ptr.argmax(1) * (ptr.sum(1) > 0.5) - (ptr.sum(1) < 0.5), h[:, ON] > 0.5

    readout = lambda s: "".join(io["charset"][i] for i in (s @ net["W_out"] + net["b_out"]).argmax(-1))
    rec, digests, grids = [read(h)], [hashlib.sha256(h.tobytes()).digest()], [readout(h)]
    worst = 0.0
    for t in range(1, R + 1):
        h = run.layer(net["layers"][0], h, None)
        body = h[:, 4:]
        worst = max(worst, float(np.abs(body - np.round(body)).max()))
        assert np.isin(body, (0.0, 1.0)).all(), f"a stored value is not exactly 0 or 1 after loop {t}"
        assert (h[:, SCRATCH:] == 0).all()
        rec.append(read(h))
        digests.append(hashlib.sha256(h.tobytes()).digest())
        grids.append(readout(h))

    phase, ptr, on = (np.array(x) for x in zip(*rec))             # (R + 1) x n each
    first = lambda m: np.where(m.any(0), m.argmax(0), -1)           # first loop at which a cell's flag is set
    reach_loop, step_loop, on_loop = first(phase >= 0), first(ptr >= 0), first(on)
    final_phase, final_ptr = phase[R], ptr[R]
    t = np.arange(R + 1)[:, None]                                   # the events rebuild the state exactly, every loop
    assert (np.where((reach_loop >= 0) & (reach_loop <= t), final_phase, -1) == phase).all()
    assert (np.where((step_loop >= 0) & (step_loop <= t), final_ptr, -1) == ptr).all()
    assert (((on_loop >= 0) & (on_loop <= t)) == on).all()

    last_change = max(u for u in range(1, R + 1) if grids[u] != grids[u - 1])
    same_from = R
    while same_from > 0 and digests[same_from - 1] == digests[R]:
        same_from -= 1
    gen = "".join(io["charset"][i] for i in run.generate(net, toks))
    assert grids[R] == answer == gen, "final grid differs from the dataset's answer or from run.generate"
    reach_loop, step_loop, on_loop = (x.tolist() for x in (reach_loop, step_loop, on_loop))
    path = [i for i in range(n) if on_loop[i] >= 0]
    out = {
        "about": "Real internal state of the public Maze-Hard network (weights/run.py, networks/maze.npz) on test maze 1, "
                 "read from the residual stream after every loop.",
        "side": side, "question": question, "answer": answer, "loops": R, "start": S, "goal": G,
        "reach_loop": reach_loop, "phase": [int(x) for x in final_phase], "step_loop": step_loop,
        "step": [DIRS[x] if x >= 0 else "" for x in final_ptr], "on_loop": on_loop,
        "start_reached_at": reach_loop[S], "readout_last_change": last_change, "state_unchanged_from": same_from,
        "max_distance_from_whole_number": worst, "path_cells": len(path), "exact_match": True,
    }
    (HERE / "maze_inside_trace.json").write_text(json.dumps(out))
    print(f"side {side}, R {R}, start reached at loop {reach_loop[S]}, readout last changes at loop {last_change}, "
          f"state unchanged from loop {same_from}; every stored value exactly 0 or 1 after every loop "
          f"(max gap {worst}); path cells incl. S and G: {len(path)}; matches the dataset and run.generate")


if __name__ == "__main__":
    main()
