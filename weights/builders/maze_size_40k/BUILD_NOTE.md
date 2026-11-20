# Maze-Hard looped transformer: build note

Network folder: `work/net/` (`network.json`, `weights.npz`, `tokenizer.py`), written by
`python3 -I work/write_weights.py`. The tokenizer is a hand-written file in `work/net/`.

## Convention (from DEV analysis)
In all 1,000 DEV answers the path is a shortest S→G path. The cells marked `o` are the path cells strictly between S
and G. Ties are broken by walking from S and, at each cell, taking the first neighbour in the order
**Down, Left, Right, Up** that is one step closer to G. This rule matches 1000 of 1000 DEV answers.

## What the network computes
- **Tokenizer** (generic): one token per character, with five fields: the two low hex nibbles of the code point,
  the position's two base-32 digits, and a first-token flag.
- **Block**: layer 0 has 4 attention heads (dk 4, dv 3, no mask) and an MLP with d_ff 25. Layer 1 is an MLP only,
  with d_ff 4. d_model is 40.
- **Neighbour heads.** Each head scores keys by −β((lo_k−lo_t)² + (hi_k−hi_t)²) (β = 40), where (lo_t, hi_t) are
  the digits of p−31, p+31, p−1 or p+1 with the carry written out. The score is largest exactly at the neighbour.
  When the neighbour is off the grid, a flag taken from the embedding adds a large bonus to newline keys, so the
  head reads newlines: never reached and never on the path. `work/check_heads.py` checks every head at every
  position.
- **BFS from G**:
  - `r` is a reached flag, and the MLP propagates it through open cells.
  - `T` grows by 1 − r every loop. A reached cell therefore holds its distance to G, and every other token holds
    the loop count.
- **Pointers.** Once a cell is reached, `C_dir = [T − T_neighbour ≥ 1]`.
- **Path.**
  - `P` starts at 1 on S.
  - Layer 1 exports `E_dir` (P and the first C in the order D, L, R, U).
  - A cell joins the path when a neighbour's E points at it.
- **Readout**: walls, newlines, S and G are copied. An open cell writes `o` if P = 1, and ` ` otherwise.
- **Stability.** Every update is monotone, idempotent and snapped to exact 0/1 with margins, so extra loops change
  nothing.

## Numbers
- Weights: 8,876 in total, 428 of them nonzero (`tools/params.py`). The limit is 40,000.
- R rule: constant, R = 500 (`{"const": 500, "lin": 0}`). The path needs about 2·d(S,G) + 2 loops, so R = 500
  covers S→G distances up to about 249. The largest distance in DEV is 146 and the mean is 114.
- FLOPs per prompt (n = 929, R = 500): 3.73 × 10¹⁰. The cap is 10¹⁴.

## Tests on my own inputs (random mazes I generated, not DEV), on `numpy`
With the final weights (only R differs):
- 22 of 22 random mazes (wall densities 0.3 to 0.4) solved at R = 400.
- 9 of 10 edge mazes solved at R = 400: S/G in corners, open grids with many ties, and a 231-step snake. The
  snake was the miss, because it needs about 464 loops; at R = 500 it is solved.
- Earlier versions also solved 17 of 17 other random mazes. Two edge bugs (the up flag at p = 31 and the left
  flag at p = 0) were found and fixed with the edge set and `check_heads.py`.
- A 2-maze check on `numpy32` also matches. Torch is not installed here, so `torch64` was not run.

## DEV results
**1000 of 1000 DEV problems solved.** This took 4 counted runs of the 20 allowed (`ledger/dev_ledger.jsonl`), all on
`numpy` through the orchestrator's CPU server, over disjoint sets of problems:
- Run 1: 100 problems, the 20 with the longest S→G paths (up to 146) plus 80 spaced evenly through the file
  (`work/ids100.txt`). 100 of 100 solved.
- Run 2: 250 problems (`work/ids250.txt`). 250 of 250 solved.
- Run 3: 300 problems (`work/ids300.txt`). 300 of 300 solved.
- Run 4: the remaining 350 problems (`work/ids350.txt`). 350 of 350 solved.
- An earlier full 1,000-problem request returned nothing before my waiting process hit the 2-hour limit on
  background commands. It was never scored or counted, and its request is still in `bridge/`.
- The DEV analysis scripts (`work/analyze*.py`) only read the DEV file; they never run the network. They confirmed
  the shortest-path D/L/R/U rule on all 1,000 answers.

## Worries
- A maze whose S→G distance is over about 249 would not finish in R = 500 loops. That is far above the DEV maximum
  of 146.
- `torch64` was not run here because torch is not installed. Every decision is snapped with margins of at least
  0.25, and the attention errors are about e^-40, so float64 on the GPU should match numpy. `numpy32` also matched
  on the tests.
