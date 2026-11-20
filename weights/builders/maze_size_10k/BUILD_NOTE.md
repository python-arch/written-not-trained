# Maze-Hard: a hand-written looped transformer

Network folder: `work/net/` (written by `python3 -I work/write_weights.py`).

## What the network computes

**Convention, worked out from DEV.** The answer marks the cells of the parent path of a BFS from S. Neighbours are
expanded in the order Down, Left, Right, Up, and each cell's parent is the cell that first discovers it. S and G keep
their letters, every other path cell becomes `o`, and nothing else changes. This is the same as the
lexicographically smallest shortest path from S (D < L < R < U). A pure-Python version of the local algorithm below
(`work/ref.py`) reproduces all 1000 DEV answers.

**Tokenizer** (generic): one token per character, 10 fields:
- bits 0–6 of the code point;
- a flag for code point ≥ 128;
- line number mod 64;
- column mod 64.

**Block** (one layer, applied R times): d_model 20, 4 heads (dk 3, dv 2, no mask), d_ff 13.
- **Neighbour attention.** Each head attends to the token one line up, one line down, one column left or one column
  right. The score is −β·(squared line/column distance to the target), built from the line value, the column value
  and line² + column² in the residual stream, with β = 40. The margin is ≥ 40 nats per unit of squared distance.
- **Wave from G.** An open cell becomes visited the loop after one of its neighbours does. In that same step it
  latches a one-hot pointer to the first visited neighbour in the order D, L, R, U. Those neighbours are exactly the
  ones one step closer to G. This takes 4 ReLU units.
- **Path from S.** A cell not yet on the path that is S, or that receives a "send" bit pointing at it from a
  neighbour, copies its pointer into its own send bits (4 units). The send bits are therefore set along the pointer
  chain S → G.
- **Scratch clearing.** Attention writes into 5 scratch dims, which 5 more units reset every loop. All values there
  are ≥ 0, so relu(x) = x and the reset is exact.
- **Readout.** `o` where a send bit is set; otherwise the input character, with S and G winning over `o`. Every
  position writes exactly one character, newlines included. All logits are small integers or halves, and the
  smallest winning margin is 0.5 on every position checked.

Outside the grid, a neighbour head falls back to the cell itself. That is harmless: a cell's own visited bit is 0
whenever it is read for a pointer, and its own send bit can only be 1 once it is already on the path.

## Numbers

- **Weights:** 4,380 total (`tools/params.py`), 392 nonzero. Limit 10,000.
- **R rule:** R = 480 (constant). A path of length d needs exactly 2d loops (checked), so 480 covers d ≤ 240; DEV
  has d from 109 to 146. After convergence every bit is latched, so extra loops change nothing.
- **FLOPs per prompt:** 2.61 × 10¹⁰ (n = 929 tokens). Cap 10¹⁴.

## DEV results

One counted DEV run (run 1 of 20): all 1000 DEV mazes on `numpy`, solved 1000/1000 (mean 1.0000). It was
evaluated on the orchestrator's CPU server through `tools/dev_run.py`. The log is in `work/dev_run1.log`. No other
DEV runs were used.

## Checks on my own inputs (not counted)

`work/gen_own.py` generates random 30 × 30 mazes, with S/G on borders and corners in some of them; answers come from
`work/ref.py`. 24 mazes × {numpy, numpy32}: 48/48 solved, smallest logit margin 0.5.

## Worries

- **Tokenizer.** The line/column fields are a generic text feature (like line:col in an editor), but they are what
  make 2-D neighbour attention cheap. Lines or columns past 63 wrap, so the tokenizer works on any text; the network
  is only meaningful on short lines.
- **Long paths.** A TEST maze with a shortest path longer than 240 steps would come out unsolved. This is far above
  the DEV maximum of 146.
- **torch64.** It was not run (no GPU). It computes the same float64 operations as `numpy`, and every decision has a
  margin ≥ 0.5 in the logits and ~e⁻⁴⁰ attention leakage, so I expect identical output.
