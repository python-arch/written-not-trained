# GraphWalks BFS: hand-written looped transformer

Network folder: `work/net/` (`network.json`, `weights.npz`, `tokenizer.py`). Weights are written by
`python3 -I work/write_weights.py`, which reproduces the hashes used in both DEV runs
(network.json 0523244d0271, weights.npz e5a319b4d9ab, tokenizer.py 5b6cf2ff69c2).

## Tokenizer (generic)
- The tokenizer encodes text as UTF-8 bytes and splits it into lines at `\n`. Each line is cut into 4-byte chunks (an empty line gives one empty chunk). One token per chunk.
- Each token has 87 fields:
  - the bytes at stream offsets −24…+59 around the chunk start (byte+1; 0 means outside the text);
  - the chunk's index within its line (capped at 127);
  - an is-first-token flag and an is-last-token flag.
- Nothing in it depends on edges, arrows, IDs or the question format.

## What the network computes
d_model 352. The block has two layers: one with a single attention head (dk 21, dv 2, no mask) plus an MLP of 811 units, then an MLP-only layer of 12 units.

- **Embedding (linear in the window bytes).**
  - For each window offset, a 2-d circle code of the hex digit there (cos/sin of 2πv/16).
  - One-hot hex digits at offsets −8…1.
  - Template-match counters: one per chunk index 0–5 for the edge-line form `<10 hex> -> <10 hex>\n`, and one for the BFS operation line seen from its chunk 6.
  - Digit values of the depth N, chunk one-hots, the first/last flags, and a constant.
- **Layer-1 MLP.** Its updates are idempotent "set" updates.
  - It builds the canonical source and target codes of the token's own edge line (it selects window offsets by chunk index), the hex digits that this token would write, and validity flags.
  - It sets the operation token as a virtual edge "→ X", marked reached, with countdown N+1, where N is parsed from its digits.
  - It computes the newly-reached gate G and E = |countdown|.
- **BFS head.** The query is the line's source-node code. The key is the target-node code plus γ·reached; every non-edge token is a constant "sink" key. The value is (reached, countdown).
  - A line becomes reached when some line whose target is its source is already reached.
  - It then takes the parent's countdown − 1, so countdown = N − dist(source). Scores: β=600, γ=25, sink margin δ=12. The exact-match gap is ≥45 logits.
- **Layer-2 MLP.**
  - Applies `reached += G` and `countdown := parent − 1` when G fires, and clears the scratch dims.
  - Sets the answer flag to reached ∧ countdown = 0.
- **Readout (alphabet of 260 strings).** The alphabet is `""`, `Final Answer: [`, `]`, `,` and the 256 hex pairs.
  - The first token writes `Final Answer: [` and the last token writes `]`.
  - On each edge line whose source is at exactly depth N, chunks 0–4 write the source ID two hex digits at a time and chunk 5 writes `,`.
  - A node with several out-edges is written once per edge. Since F1 compares sets, the duplicates do not change the score.

## Size and compute
- **Parameters:** 8,334,331 in total, 7,777 nonzero (stored sparse).
- **R rule:** R = 400 (const 400, lin 0, log2 0). Answers are exact for depth N ≤ R − 3 = 397.
- **FLOPs at 1,800,000 characters:** about 432k tokens, giving ≈ 4.0e15 (cap 5e15). One 1.79M-character own prompt counted 3.985e15 and took 921 s on the GPU.

## DEV results (2 of 20 runs used)
1. **DEV run 1** (GPU `torch32`): file A, all 350 BFS problems, mean F1 **1.0000**.
2. **DEV run 2** (GPU `torch32`): 7 own inputs, none built from DEV, mean F1 **1.0000**. Five were between 1.0M and 1.79M characters, including a long chain asked at depth 395 and an answer of 4,238 nodes.
- No local runs were counted. Local numpy/numpy32 testing used only my own random graphs.
- On those, depths 396 and 397 come out right; depth 398 fails, which is the expected R limit.

## Worries
- **Format.** The network relies on the DEV format: 10-character hex IDs and the exact BFS operation sentence with the ID at column 24. Every DEV prompt (A and B) matches. A different template or ID length would make the output `Final Answer: []`.
- **Nodes with no out-edges.** An answer node is written from its own out-edge lines. Every node in DEV has out-degree ≥ 1; a node with no out-edges at the target depth would be missed.
- **Depths above 397.** These are answered with an empty list. That is wrong only if the true answer at such a depth is non-empty, which looks rare; the largest depth asked in DEV is 39.
- **Runtime.** 400 loops make the longest prompts take about 15 minutes each on the GPU.
