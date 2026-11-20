# Maze-Hard looped transformer — build note

## What the network computes
Network folder: `work/net/` (`network.json`, `weights.npz`, `tokenizer.py`). Every weight is written by
`work/write_weights.py`. Nothing is trained.

**Convention, worked out from DEV.** The answer marks with `o` the cells strictly between S and G on one shortest
path. Of all shortest paths it picks the lexicographically smallest move sequence from S, with moves ordered
Down < Left < Right < Up. That is the same as walking greedily from S and always taking the first neighbour, in that
order, that is one step closer to G. This reproduces all 1000 DEV answers when checked with a plain reference
solver (no network).

**Tokenizer (generic).** One token per UTF-8 byte. It has 11 fields: the byte's 8 bits, then the byte's position
mod 9, 10 and 11, a residue-number position code.

**Block.** d_model = 28. There are 2 layers, run R times.
- *Layer 0*: one causal head whose score is a bonus on newline tokens, plus a 2-unit MLP. It produces three
  boundary flags:
  - "no newline before me", which is true in row 0;
  - "I am in the last row", from the mean of a fixed phase functional over the newlines seen so far. This mean is
    below 0 above row 29 and above 0 in it, which is a fixed property of the 30×31 layout and is checked in the
    script;
  - "I am the last token", from one MLP unit on the position phases.
- *Layer 1*: four neighbour heads (down +31, left −1, right +1, up −31) plus a 10-unit MLP.
  - Each head's query is the position-phase code rotated by its offset. That makes the score
    Σ_m w_m cos(2π(j−i−δ)/m) times β=900, which peaks only at j = i+δ. The kernel margin is 0.063, so everything
    off the target gets less than 1e-22 of the attention.
  - Where the neighbour doesn't exist, the matching flag adds a large bonus on newline tokens. Newlines carry
    zero state, so they act as walls. The left head is causal, so token 0 reads itself, which does no harm.
  - Each head reads two things from its neighbour: whether it is reached, and its mark-out toward me. They add
    into accumulators.
- *MLP of layer 1*, all set-once with dead zones so attention leakage cannot grow:
  - **reach** (2 units): an open cell becomes reached one loop after a neighbour. G starts reached.
  - **pointer** (4 units): fixed at the loop the cell is reached. It is the first direction (D, L, R, U) whose
    neighbour was reached exactly one loop earlier.
  - **mark-out** (4 units): set when the cell is marked (incoming ≥ 1, and S starts marked) and its pointer points
    that way. Each loop the mark moves one cell along the pointer chain from S.
- **Readout**:
  - `o` = 4 × incoming marks, against a bias of 1 for space;
  - `S` and `G` come from a ±1 "S/G" dimension (×1e5);
  - `#` and `\n` come from their character flags (×1e5).

## Size, loops, compute
- Weights: **4,359** (nonzero 259); limit 5,000 (`tools/params.py`).
- R rule: `R = ⌈n/2⌉` (`{"const":0,"lin":1,"div":2,"log2":0}`), which gives R = 465 for the 929-token maze.
  A maze needs 2·DG(S) loops, where DG(S) is the S→G distance. So it handles DG(S) ≤ 232. I checked this
  boundary on my own serpentine mazes: 232 is solved, 233 is not. DEV's largest DG(S) is 146.
- FLOPs per prompt: 4.31e10 (cap 1e14).

## Tests and DEV results
- Own mazes (my own generator, not counted): 193 random mazes plus edge cases (S/G in corners and on every border),
  and 3 long serpentines. All were solved except the deliberate DG = 233 over-length case. An early version failed
  4 of the first 40: newline token 30 has no up-neighbour and no sink, so its readout was noisy. I made the `\n`
  and `#` logits dominant, and every own maze since has passed.
- DEV: **1000 / 1000 solved** (mean 1.0000) in DEV run 1, with all 1000 DEV mazes on backend `numpy`. **1 of 20 runs used.**
  The network hashes were network.json ae8518e95fb6, weights.npz 6288ddb683aa and tokenizer.py 1431cc2eefe5.

## What worried me
- The boundary handling depends on the fixed 30×31 layout: the newline positions, and the "last row" functional
  (layout margin 0.025, rescaled to ±1). It is correct for every maze of this format, but it is not a general grid
  solver.
- The kernel and the flags use large constants (β=900, sink bonus up to ~1e8, gates 1e4). That is fine in float64,
  which is what `numpy` and `torch64` use. I did not design for float32.
- `tools/dev_run.py` now sends numpy runs to the "orchestrator's CPU server" through `bridge/`. Its docstring
  describes this as a harness change. I used it as given; only prompts are sent.
