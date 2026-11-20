# BUILD NOTE: strongly_connected_components

## What the network computes
The dataset's id rule, checked against all 996 DEV answers with a plain-Python reference (`work/ref.py`): each node gets
the id of the member of its SCC that a DFS discovers first. The DFS starts from nodes 0, 1, … in order and tries
neighbours in increasing index. This is what Kosaraju's second pass in CLRS gives: the root is the member with the
latest finish time.

- **Tokenizer** (`net/tokenizer.py`): one token per character.
  - Field 0: the character code, clipped to 127.
  - Field 1: the position, clipped to 4095.
  - Nothing else; it is fully generic.
- **Block**: 7 layers, the same in every loop. Every state dimension is *set* each loop (the MLP adds new − old), and
  attention writes into scratch dimensions that the next MLP clears. All booleans pass through flat steps
  `relu(3z−1) − relu(3z−2)`, so float error is cut back to an exact 0 or 1 each loop and never builds up.
  1. **Parsing.** Attention finds the first `[`, the first `]` and the last `[` at or before each token. The MLP turns
     these into n (one-hot), each token's offset from row 0, and the column one-hot / `1`-bit.
  2. **MLP only.** Row one-hot, from offset = u·(2n+3) gated by one-hot n; the `1`-bit by row; an in-matrix flag;
     the mask of valid nodes.
  3. **Rows and columns.** Every token of row u gathers row u and column u of A. The MLP binarises them and sets the
     flags for node, first node and last node.
  4. **Reachability.** Squaring of forward reach F and backward reach B (log-many loops). AU = A[u] ∧ ¬visited.
  5. **One DFS step per loop** (global state copied in every token). Visit the lowest unvisited child of cur;
     otherwise backtrack to parent[cur]; if there is no cur, restart at the lowest unvisited node.
  6. **Recording.** The newly visited node records its parent and its discovery index. SCC set S = F ∧ B.
  7. **Answer.** Attention picks the member of S with the smallest discovery index (score λ·[member] − μ·disc). The
     MLP sets the output indicators.
- **Readout**: the row-opening `[` of row u writes `"[id "` (u = 0), `"id "` or `"id]"` (u = n−1). Every other
  position writes `""`. The alphabet is `""` plus 72 strings for ids 0–23.
- **Capacity**: up to N = 24 nodes. TEST goes to 22.

## Parameters
- d_model is 641; d_ff per layer is [372, 2498, 247, 264, 240, 1296, 240].
- 9,785,608 parameters in total (dense count, mostly the 4096-row position table); 40,061 are nonzero.

## R rule
R = 12 + ⌈n_tokens / 16⌉. The network needs R ≥ 2·nodes + 2: one DFS step per loop, with each node visited once and
backtracked once, plus a few loops for parsing and for the reachability squaring to settle. Since
n_tokens ≥ 2·nodes² + 3·nodes + 70, this holds for every size.

## FLOPs per prompt (cap 1e14)

| nodes | tokens | R | FLOPs |
|---|---|---|---|
| 4 | 121 | 20 | 3.4e10 |
| 5 | 144 | 21 | 4.3e10 |
| 10 | 319 | 32 | 1.5e11 |
| 11 | 367 | 35 | 1.8e11 |
| 12 | 419 | 39 | 2.3e11 |
| 15 | 599 | 50 | 4.3e11 |
| 22 | 1159 | 85 | 1.5e12 |
| 24 | 1355 | 97 | 2.0e12 |

## Results
- **Own inputs** (made with `work/gen.py`, not counted): 140/140 solved with numpy float64. The sizes run from 4 to 24
  nodes, with random densities from 0.05 to 0.5. They also include chains, cycles, complete graphs and reversed
  triangular DAGs (deep DFS stacks).
- **DEV**: 1 run used of 20 (numpy float64, all 996 problems): **996/996 solved** (mean solved 1.0000). The log is
  `work/dev_run1.log`.

## Worries
- Attention margins are wide: score gaps of at least 40, giving leaks below 1e-15. Gate constants reach 1e4 against
  positions up to about 1.4e3. numpy float64 and torch64 do the same exact softmax, so they should agree.
- I could not run torch64 here (there is no torch and no GPU).
- The network relies on the exact prompt format: `A: [[`, rows of `0`/`1` separated by single spaces, and
  `, initial_trace: [`. It handles at most 24 nodes and prompts up to 4095 characters.
