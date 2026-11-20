# mst_prim looped transformer: build note

## What the network computes
Tokenizer (`net/tokenizer.py`, generic): each decimal number literal `[0-9]+(\.[0-9]+)?` is one token and every
other character is its own token. Each token has 4 fields: char code (number tokens get 256), integer part (capped
at 999), the first 3 fractional digits as an integer, and the position (capped at 8191).

d_model = 43, a block of 5 layers (11 heads, every dk = 4, so 1/√dk = 1/2 exactly), no training. Every layer
"sets" its outputs: its MLP adds new − old and clears the scratch dims the heads wrote. Running the block again
therefore leaves the parse unchanged, and each loop advances only the Prim state.
- **P1**: global or causal argmax-position heads find the first `[`, the last `[`, the most recent `[` and the
  last `]`. From these the MLP flags each token: matrix entry, node slot (the numbers of `initial_trace`), the
  start-node token, and which tokens echo `[`, ` ` or `]` in the output.
- **P2**: it reads s. Causal uniform heads give 1/(row+2) and 1/(col+2), and 40 fixed relu-step thresholds turn
  them into the integers u, v and v². Weights become integer milli-units w = 1000·int + frac. The MLP also sets
  `edge = w ≥ 1` and `start = (slot id == s)`.
- **A**: each entry (u,v) fetches age[u] and age[v] from the node slots using a quadratic-match score. Then
  `elig = edge ∧ age[u] > age[v]`. age is the number of loops a node has been marked, so u was extracted before v,
  or v is still unmarked.
- **B**: slot v takes the argmin over eligible entries in its column of 128·w − age[u], in one sharp softmax. That
  gives key[v] and parent[v]: the minimum weight, ties going to the earliest-extracted u. This is exactly Prim's
  strict-< relaxation. The layer then sets `cand = slot ∧ unmarked ∧ (has key ∨ start)`, `cost = 64·key + id`, and
  `out = (marked ∧ ¬start) ? parent : id`.
- **C**: every slot attends to the candidate with minimum cost, which is one EXTRACT-MIN with ties to the lowest
  index. The chosen slot gets mark += 1, and every marked node's age goes up by 1.
- **Readout**: at slots the logit of "k" is 2k·out − k² (plus a gate), which writes the parent number. The trace
  bracket and spaces echo `[`, ` `, `]`, and every other position writes "". The output is, for example,
  `[5 4 2 5 3 2 3 4 1 7]`. pi[s] = s and unreachable nodes give pi[v] = v.

## Size
Parameters: 477,077 total, 13,515 nonzero (stored sparse).

## R rule
R = 2 + ⌈n/24⌉ (n = tokens). A size-N problem needs N + 1 loops: one extraction per loop, then one to write the
outputs. Extra loops do nothing.

| nodes | tokens | R | FLOPs |
|---|---|---|---|
| 4 | 102 | 7 | 4.92e7 |
| 5 | 125 | 8 | 7.28e7 |
| 10 | 300 | 15 | 4.61e8 |
| 11 | 347 | 17 | 6.51e8 |
| 12 | 398 | 19 | 9.00e8 |
| 15 | 575 | 26 | 2.23e9 |
| 19 | 867 | 39 | 6.73e9 |
| 23 | 1223 | 53 | 1.69e10 |
| 28 | 1758 | 76 | 4.69e10 |

All sizes are far below the 1e14 cap.

## DEV results
**1 DEV run used of 20** (run 1, all problems, numpy float64): **997 / 999 solved (0.998)**. The DEV file holds
999 problems, 111 per size, not the 1,000 the brief states. The two failures, ids 58 (19 nodes) and 190 (23
nodes), are exactly the extraction-tie cases predicted below.
After that run I changed only `tokenizer.py`: it now returns an (n, 4) numpy array, so empty text gives shape (0, 4)
and not an empty 1-D list. The tokens are identical on all non-empty text, and the own tests were rerun and still all
pass. network.json and weights.npz are unchanged, with the same hashes as in the ledger. The runner itself still
can't take empty text (its range check reduces over zero tokens). That is a frame limit, not the network's.

Before any DEV run I tested on my own generated problems (`work/gen.py`, `work/gen2.py`; reference Prim in
`work/ref.py`) and solved all 255 of them. These included sparse graphs, disconnected graphs, weight 1, and heavy
weight ties.

## Worries
- **Tie-breaking in extraction.** The dataset's generator breaks ties among equal keys with `np.argsort(...)[0]`.
  For more than 16 nodes, that is introsort and not the lowest index. Lowest-index tie-breaking, which the network
  implements, disagrees with the DEV answers on exactly 2/1000 problems (ids 58 and 190). I did not try to emulate
  numpy's quicksort, so expect about 0.2% loss on TEST from this.
- **float64 only.** Attention scores reach about 1e11 with integer margins of at least 64, which is fine in
  float64 (numpy and torch64 agree up to rounding far below the margins). The network does **not** work in float32
  (numpy32 fails), but TEST uses torch64.
- Sizes are general up to about 50 nodes (R < 128 is needed for the age tie-break, and ids < 64 for the alphabet).
  Matrix rows and columns are counted up to 40.
- Housekeeping: I once accidentally listed the shared `$TMPDIR`, which held other folders' names. I read nothing
  there, and afterwards kept my scratch files in `work/tmp`.
