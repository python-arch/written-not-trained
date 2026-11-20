# Networks written by fresh builder sessions

Each network here was written by a separate headless Claude Code session (Claude Opus 5.5), the builder. It started
from a plain brief (`prompts/04`, `05`, `11`, `13` for the maze size-limit builds and `04b` for the Sudoku loop-limit build) and nothing else. No weight was trained: each one is set by the build's
`write_weights.py`, which reads no data. Every build was frozen before TEST, run once on TEST, and checked by a fresh
auditor (all PASS WITH NOTES). The files are byte for byte the frozen ones: `SHA256SUMS` in each folder, matching the
hashes recorded before TEST.

All networks use the same frame: embed the prompt, run one shared block R times, then read out. `runner/` is that
frame, the same files every builder had (`runner/README.md` gives the network format and exactly what is computed).

| Folder | Task | TEST result |
|---|---|---|
| `sudoku_extreme/` | Sudoku-Extreme | 141,040 / 141,040 puzzles finished, all solved (first 10,000: 10,000 / 10,000; the run was stopped to free the GPU) |
| `graphwalks/` | GraphWalks BFS, 256k–1M-token prompts | 200 problems, mean F1 1.000 (current card grader) |
| `clrs_scc/` | CLRS-Text `strongly_connected_components` | 9,300 / 9,300 (every size, 4–22 nodes) |
| `clrs_mst_prim/` | CLRS-Text `mst_prim` | 16,269 / 16,400 exact match; a minimum spanning tree on all 16,400 (the 131 misses are equal-weight ties) |
| `clrs_lcs_length/` | CLRS-Text `lcs_length` | 6,000 / 6,000 |
| `satlib/` | SATLIB random 3-SAT (uf20-91, uf50-218, uuf50-218) | 2,700 / 2,700, no wrong answer and no timeout |
| `planbench/` | PlanBench task 1: Blocksworld, Mystery Blocksworld, Logistics (one network each) | 400 / 400, 400 / 400, 200 / 200 valid plans |
| `maze_size_1k/` | Maze-Hard (30 × 30), written under a 1,000-weight limit: 928 weights | 1,000 / 1,000 |
| `maze_size_5k/` | Maze-Hard (30 × 30), written under a 5,000-weight limit: 4,359 weights | 1,000 / 1,000 |
| `maze_size_10k/` | Maze-Hard (30 × 30), written under a 10,000-weight limit: 4,380 weights | 1,000 / 1,000 |
| `maze_size_20k/` | Maze-Hard (30 × 30), written under a 20,000-weight limit: 7,682 weights | 1,000 / 1,000 |
| `maze_size_40k/` | Maze-Hard (30 × 30), written under a 40,000-weight limit: 8,876 weights | 1,000 / 1,000 |
| `sudoku_loop_limit/` | Sudoku-Extreme, written under a limit of 1,000 loops: R = 1,000, a 60-layer block | first 10,000: 10,000 / 10,000 |

## Run

Python 3 with numpy (torch for the GPU backends):

```
cd weights/builders
python3 runner/looped.py clrs_mst_prim/net clrs_mst_prim/example_dev_prompt.txt
python3 runner/looped.py planbench/net_logistics prompt.txt --backend numpy
```

- The first argument is a network folder (`network.json`, `weights.npz`, `tokenizer.py`); the second, a text file holding
  one prompt, exactly as the dataset gives it to a language model.
- It prints the network's answer, plus a line on stderr: tokens, R and FLOPs.
- Backends: `numpy` (float64, the default), `numpy32`, `torch32`, `torch64`. TEST used `torch64` for CLRS-Text and the maze size-limit builds, and
  `torch32` for the others.
- `example_dev_prompt.txt` in each folder is the build's first DEV prompt.
- Without a `build.json` next to `runner/`, the FLOP cap defaults to 5e15 per prompt. Each build's own cap is in its
  `build.json`.

## In each folder

- **Network folder(s):** `net/`, or for PlanBench `net_blocksworld/`, `net_mystery_blocksworld/` and `net_logistics/`.
- **`write_weights.py`:** the builder's code that writes every weight. Run it as `python3 -I write_weights.py` from a
  folder laid out like the builder's `work/`.
- **`BUILD_NOTE.md`:** the builder's note:
  - what the network computes;
  - parameters;
  - the R rule;
  - FLOPs per size;
  - DEV results;
  - what worried it.
- **`grader.py`:** the build's scoring (with `domains.json` for PlanBench).
- **`build.json`:** the build's FLOP cap and DEV budget.
- **`sim.py` (SATLIB only):** the builder's step-level simulator, used for the search-step counts.
