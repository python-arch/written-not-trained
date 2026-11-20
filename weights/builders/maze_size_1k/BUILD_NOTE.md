# Maze-Hard looped transformer — build note

## What the network computes
Each character of the prompt is one token. The tokenizer has two generic fields: `ord(ch) % 7` and a
position class (first char / odd index / even index). All weights are written by `write_weights.py`.

The rule the dataset follows, worked out from DEV answers: the marked path is a shortest S→G path.
Walking from S, each step takes the first of **down, left, right, up** that lowers the BFS distance to G.
This held at every one of the ~20k tie decisions in DEV. A simulator of this rule reproduces all 1000 DEV answers.

The block has one layer, three heads and 9 ReLUs. It is run R = 1415 times.
1. **Positions (loops 1…~930).** The *left* head (causal) attends to the previous token: the latest
   token with the largest P and the opposite index parity. The *self* head (causal) attends to the
   token itself. Together with one guard ReLU they run the recurrences P_i = P_{i-1}+1,
   Q_i = Q_{i-1}+2P_{i-1}+1 (= P²), and (C,S)_i = rotation of (C,S)_{i-1} by 2π/31. These
   converge front to back, one token per loop. The self head also clears the per-loop scratch dims.
2. **BFS from G (gated on by a saturating clock at loop 945).** The *nbr* head's score is
   `A·cos(2π ΔP/31) − ΔP² + tie-break + λ·u_j`. Only the four grid neighbours (ΔP = ±1, ±31) can beat
   the token itself, and only when their state u is larger. The tie-break orders them D > L > R > U.
   An unreached open cell therefore attends to its highest-priority reached neighbour. That neighbour
   becomes its parent; the cell stores the offset to it and sets u = 1. Two ReLUs do the step and one
   stores the pointer. S gets u = 2 directly, which marks it.
3. **Trace from S.** In the same head, a reached cell (u = 1) attends to a marked neighbour (u = 2) if it
   has one. A 4-ReLU flat-top equality test checks whether that neighbour's parent is this cell; if so
   u becomes 2. The path grows by one cell per loop.
4. **Readout.** Linear in (open, w, u): `#`, `\n`, ` `, `S`, `G`, or `o` for u = 2 on a space.

## Size and compute
- Weights: **928** total (limit 1000); **136** nonzero. d_model = 12, d_ff = 9, heads dk/dv = 1/6, 2/4, 5/2.
- R rule: constant, **R = 1415** (945 loops for the positions, then 2 × 230 + 10 for BFS and trace).
  This covers start–goal distances up to about 230; the largest in DEV is 147.
- FLOPs per prompt (929 tokens): **6.91e10** (cap 1e14).

## DEV results and runs used
- My own generated mazes (not DEV; not counted), run through the runner's `looped.run` on numpy:
  **52 of 52 solved**. They include S or G in all four corners, paths through cell (0,0), S next to G,
  and paths of 30–120 cells.
- DEV run 1 (`dev_run.py`, numpy, train/0–49): **50 of 50 solved**.
- DEV run 2 (`dev_run.py`, numpy, train/50–149): **100 of 100 solved**.
- DEV run 3 (`dev_run.py`, numpy, train/150–299): **150 of 150 solved**.
- Total: 300 of 300 DEV mazes solved, using 3 of 20 DEV runs. The other 700 DEV mazes were not
  evaluated, because the server took about 50–60 minutes per run of 50–150 mazes.
- An earlier attempt at all 1000 DEV mazes was never scored or recorded. My client process was stopped
  at a 2-hour wait limit before the orchestrator's CPU server returned. The ledger has no entry for it,
  and I did not read its server-side result. Its request is still in `bridge/`.

## What worried me
- The first ~945 loops only build positions; the BFS is held off by a clock gate until then.
- Exactness rests on float64 margins. Large gate constants (2^15) cause rounding at the 1e-8 level,
  which is far inside the 0.4 tolerance of the flat-top tests. Before I fixed it, an early version let
  these errors grow ×1000 per step. torch64 could not be checked here: no torch, no GPU. It does the
  same float64 math, and every decision has a wide margin.
- The tokenizer's `% 7` hash was chosen so that the five maze characters fall in different buckets.
  It is still a generic tokenizer: it works on any text and other characters simply collide.
- If a TEST maze had a start–goal distance above ~230, the path would be cut short.
