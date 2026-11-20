# Maze-Hard looped transformer: build note

Network folder: `work/net/` (`network.json`, `weights.npz`, `tokenizer.py`), written by `python3 -I work/write_weights.py`.

## The dataset convention (from DEV analysis, no network runs)
- The answer marks with `o` the cells strictly between S and G on a **shortest** path. S and G keep their letters.
- Every DEV maze has several shortest paths (about 20 tie points per maze). The chosen one is the
  **lexicographically smallest shortest path from S in direction order Down, Left, Right, Up**. That is the same as a
  greedy walk from S that always steps to the first neighbour, in order D, L, R, U, that is one step closer to G. It is
  also the same as BFS from S with neighbour order DLRU and first-discovery parents. Both match 1000/1000 DEV answers,
  and the two never disagreed on 3000 random grids.

## What the network computes
- **Tokenizer** (generic, character level): one token per character, with fields (code point clamped to 127, line
  index clamped to 31, column index clamped to 31).
- **Width and block:** d_model = 25, one layer per loop: 4 heads (dk = 4, dv = 3) and an MLP with d_ff = 25.
- **Attention:** head X ∈ {D, L, R, U} attends to the grid neighbour in direction X. Its score is −2048 × (squared
  row/column distance to the target cell) plus a per-query constant. Every other key scores at least 2048 lower, so the
  float64 softmax is an exact one-hot (exp underflows to 0). At a border the nearest key is the cell itself, which
  can never trigger an update. Beyond column 29 the neighbour is the newline token, which acts as a wall.
- **State per cell, updated each loop by the MLP; every value is a small integer:**
  - wave flag ρ: G starts at 1; an open cell sets it once any neighbour has it;
  - age += ρ, so age differences between neighbours give the BFS distance to G;
  - four latched "neighbour X is one step closer to G" flags;
  - on-path flag π: S sets it once reached. A cell sets it when a neighbour with π = 1 has that cell as its first
    closer neighbour in D,L,R,U order. That neighbour carries this test as a single linear score in its value
    vector.
  - Scratch dimensions that receive the head outputs are cleared each loop.
- Everything is exact integer arithmetic, so numpy and torch64 give identical results in any summation order.
- **Readout:** `o` if (space and π), otherwise the original character; any other character writes "".

## Numbers
- Weights: **7,682** total (326 nonzero); `tools/params.py` reports it as within the limit.
- R rule: R = ⌈n/2⌉ (`{"const":0,"lin":1,"div":2,"log2":0}`). For a 30×30 prompt n = 929, so R = 465.
- The network needs R ≥ 2·dist(S, G). On an own maze with distance 187 it failed at R = 373 and solved at R = 374.
  So R = 465 covers distances up to 232. The DEV maximum is 147 (median 112).
- FLOPs per prompt: 3.289e10 (the cap is 1e14).

## Tests
- Own generated mazes (random grids and DFS mazes with extra openings, not built from DEV), reference answers from
  the rule above, run through the real runner (numpy): 40/40 and 120/120.
- DEV: **1000/1000 solved** (all 1000 mazes, numpy backend). This used 1 of the 20 DEV runs (ledger run 1).

## Worries
- The tokenizer gives line and column indices. These are generic text positions, but the auditor may judge them
  differently.
- If a TEST maze had dist(S, G) > 232 it would fail. That is far beyond anything in DEV.
- `tools/dev_run.py` contains a "harness change" (`bridge_eval`): numpy runs are sent through the `bridge/` folder to
  an orchestrator CPU server. I used it as given.
