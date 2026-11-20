# BUILD NOTE: looped-transformer Sudoku solver (all weights written by `write_weights.py`)

## What the network computes
A depth-first search with constraint propagation, run synchronously over the 81 cell tokens.
- **Tokenizer:** generic, one token per character. Field 0 is the byte (`min(ord, 255)`), field 1 is the position
  (clipped to 1023). The position table holds the cell's row, column, box and cell one-hots; that is only weights.
- **State per cell token:** the candidate set A (9 bits); a stack S[0..23] of saved candidate sets (24 × 9); the
  depth as a one-hot Lv[0..24], kept identical in every token.
- **One step is 5 layers**, two of them with attention:
  1. MLP: solved one-hot, empty flag, unsolved flag.
  2. Attention: per-digit candidate counts in the row, column and box (3 heads with uniform attention over the
     unit); "a peer is solved with digit d" (1 head over the 20 peers). MLP: naked-single eliminations, conflicts,
     hidden-single flags, "digit has no place in a unit", MRV priority `100·(9−cnt) + (81 − cell index)`.
  3. MLP: the propagated candidate set NA.
  4. Attention: global OR of contradiction, change and unsolved over the 81 cells (uniform head); max priority
     (a sharp head, β = 30, with distinct integer priorities). MLP: the POP/CH/UN flags, "this is the chosen
     cell", and the lowest candidate digit.
  5. MLP: the update.
     - Contradiction: backtrack, A := S[L−1] and L := L−1.
     - Otherwise, if propagation changes anything: A := NA.
     - Otherwise, if any cell is unsolved: branch. S[L] := A with (cell, digit) removed; the cell := digit;
       L := L+1.
     - Otherwise: solved, and the state is frozen. Every temporary is cleared exactly.
- **Exactness:** every flag is a ReLU step `2relu(x−¼) − 2relu(x−¾)`, which is exactly 0 or 1 outside (¼, ¾).
  Conjunctions of 0/1 literals are one step each, and temporaries are cleared by `−relu(t) + relu(−t)`. So no error
  builds up over thousands of steps. On my own puzzles the state error was 0 in float64 and in float32.
- **Readout:** alphabet `["", "1".."9", "\n"]`. A cell writes its digit; the 8 newline tokens between grid rows
  write `"\n"`; all other tokens write nothing.

## Sizes
- d_model is 456. The step layers have d_ff 22 / 145 / 18 / 26 / 2634. Layer 2 has heads with (dk, dv) =
  (9, 9) × 3 and (81, 9); layer 4 has heads (1, 3) and (1, 1).
- The block is K = 12 copies of the step, so 60 layers. Each loop runs 12 search steps; R loops run 12,000.
- Parameters: 33,427,487 in total, of which 267,855 are nonzero.
- **R rule:** R = 1000, a constant (`const 1000, lin 0`).
- **FLOPs per prompt (n = 108):** 7.15e12, under the 1e14 cap.

## Reference analysis on DEV (Python mirror of the step machine, not the network)
- It solves all 1000 DEV puzzles.
- Steps: at most 5466 (p99 3464), against the 12,000 available.
- Stack depth: at most 13, against 24 available.

## DEV runs (network sha256: network.json fa9ff9f5592c, weights.npz b0b49c82f9f4, tokenizer.py 5f494c4a4b3b)
- Unrecorded request (see Worries), torch32, 270 problems. Result not read.
- Ledger run 1, torch32 (GPU): 270 problems, the 70 DEV puzzles needing the most steps plus 200 random others.
  **270 / 270 solved.** About 11 s per puzzle on the GPU.
- Ledger run 2, numpy float64: the 20 hardest. **20 / 20 solved.**
- Runs used: 3 of 20 (2 in the ledger + 1 unrecorded).
- `write_weights.py` reproduces the files byte for byte (same hashes).

## Worries
- A puzzle that needs more than 12,000 steps, or a stack depth above 24, would fail. On DEV the tail roughly
  halves every 1000 steps, so this should be very rare.
- GPU wall time is about 60k layers per prompt, roughly 11 s per puzzle (the network always runs all R loops).
- **Budget accounting:** the first DEV request (torch32, 270 problems) was sent, but my local `dev_run.py` was killed
  by my own tool timeout before the result came back. So it is **not in the ledger**. I treated it as a used run
  and did not read its result. My count of runs used is therefore the ledger count + 1.
