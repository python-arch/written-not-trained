# BUILD NOTE: looped Sudoku transformer (hand-written weights)

Network folder: `work/net/` (`network.json`, `weights.npz`, `tokenizer.py`), written by `python3 -I work/write_weights.py`.

## What it computes

Tokenizer: one token per UTF-8 byte, fields (byte value, position clamped to 1023). It is generic and has no task
knowledge. The weights know the prompt layout: the position table marks the 81 cell positions with row, column, box
and cell index.

One pass of the block is one synchronous **tick** of a depth-first Sudoku solver. All cells update in parallel:

- State per cell: `cand[9]` (candidate bits), a stack of K=24 snapshots `S[k][9]`, and a depth one-hot `D` (the
  same at every position).
- **L0** (MLP): solved bits, unsolved flag, empty flag, smallest candidate `g`.
- **L1** (3 heads: row, column, box; sharp attention on a one-hot match) gives per-unit candidate counts and
  solved counts. Its MLP computes:
  - naked-single eliminations;
  - hidden singles;
  - a local contradiction count (empty cell, or a unit with no place for a digit).
- **L2** (MLP): the propagated candidates `newc`.
- **L3**: two heads.
  - A uniform head over the cells counts contradictions, changes and unsolved cells.
  - A sharp argmax head picks the guess cell: the unsolved cell with the fewest candidates, then the lowest index.

  Its MLP sets the mode flags pop / propagate / guess, and `is_sel`.
- **L4** (MLP) applies the mode:
  - **contradiction**: restore `S[D-1]`, then `D -= 1`;
  - **progress**: `cand = newc`;
  - **stall with unsolved cells**: push the current state into `S[D]`, minus the guessed digit at the chosen cell.
    The chosen cell becomes `{g}`, and `D += 1`;
  - **solved**: nothing changes. The solved grid is a fixed point.
- **Readout**: each cell writes its (single) candidate digit, and every other position writes `""`. The answer is
  the 81 digits on one line, which the grader accepts.

Each binary quantity comes from a plateau step `relu(2x-.5) - relu(2x-1.5)`. It is exactly 0 or 1 when its input is
within 0.25 of an integer. State is written with set semantics: new minus old, with old read as `relu(x)-relu(-x)`.
So float error never accumulates over loops.

## Size

- d_model 372. Five layers with d_ff 80, 238, 36, 26 and 2878, and 5 heads in total.
- Parameters: 2,973,688 in total, of which 17,014 are nonzero.

## R rule and FLOPs

- R = 40000 (constant; `{"const": 40000, "lin": 0}`). Every prompt has n = 108 tokens.
- FLOPs per prompt are 2.168e13, against a cap of 1e14.
- The simulator of the same algorithm needs at most 5,585 ticks on all 1000 DEV puzzles. Its stack depth is at most
  13, against the 24 available. The tick-count tail roughly halves per extra 1,000 ticks.

## Verification

- `work/sim/sim.py` is a plain-Python simulator of exactly this tick algorithm. It solves 1000/1000 DEV puzzles.
  This was analysis of DEV inputs, not a network run.
- Own inputs only (`work/sim/gen.py`, `unsat2.py`; no DEV grids):
  - The network state (cand, S, D) equals the simulator state at every tick, with maximum deviation 0.0, in float32
    and float64.
  - This held on 30 generated puzzles and on 2 generated unsolvable inputs that exhaust trees of 2,443 and 4,973
    ticks (605 guesses, depth 13).

- At the full R = 40000 in float32, all 30 of the same generated puzzles are solved. The solved grid stays fixed.
- With the official runner, `numpy32` and `numpy` both give the correct answer on a generated puzzle at R = 500
  (a reduced-R copy). A full-R prompt takes about 1 hour on this CPU.

## DEV results

- **DEV run 1 of 20** (GPU, `torch32`): **92/92 solved**.
  - The ids were the 40 hardest DEV puzzles by simulator ticks, up to 5,585, plus 52 at random.
  - Each puzzle took about 20–30 s on the GPU.
  - Network sha256 prefixes: network.json 1484fe3df9ab, weights.npz ba777a30dfce, tokenizer.py 5f72a8f66fd4.
- That is the only DEV run used: 1 of 20.
- Running `write_weights.py` again reproduces these weights exactly.

## Worries

- The only failure modes are a puzzle needing more than 40,000 ticks or more than 24 stack levels. On DEV the
  maxima are 5,585 and 13.
- The selection head uses scores up to about 5e4 with gaps of at least 50 between them. This is safe in float32.
  The fused kernel's summation order could only matter if a margin were near 0.25, and none is.
