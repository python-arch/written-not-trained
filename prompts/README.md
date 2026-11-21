# Prompts

These are the first messages Claude Opus 5.5 received in each experiment in the [blog post](https://python-arch.github.io/written-not-trained/). Details of our compute setup, such as machine names, accounts, paths and job ids, are removed or made generic. The messages Claude received later in each session are not included. They were fixed messages, such as "continue", the outputs of GPU runs it asked for and rule updates.

## Writing the transformers

| File | Transformer | Where in the post |
|---|---|---|
| `01_maze_hard_network_brief.txt` | Maze-Hard | Sections 1 to 3 |
| `04_sudoku_extreme_network_brief.txt` | Sudoku-Extreme | Section 5 |
| `02_tower_of_hanoi_network_brief.txt` | Tower of Hanoi | Section 5 |
| `05_graphwalks_network_brief.txt` | GraphWalks | Section 5 |
| `06_bbh_networks_brief.txt` | Boolean Expressions and Dyck Languages (BIG-Bench Hard) | Section 5 |
| `03_bbeh_web_of_lies_shuffled_objects_brief.txt` | Web of Lies and Shuffled Objects (BIG-Bench Extra Hard) | Section 5 |
| `07_graph_maps_network_brief.txt` | Graph search | Section 4 |
| `13_maze_size_limit_network_brief.txt` | Maze-Hard under a cap on the number of weights (five builds: 1,000 to 40,000) | Section 2 |
| `04b_sudoku_extreme_loop_limit_network_brief.txt` | Sudoku-Extreme with at most 1,000 loops | Section 5 |
| `11_clrs_scc_network_brief.txt` | CLRS-Text strongly connected components | Section 5 |
| `11_clrs_prim_network_brief.txt` | CLRS-Text minimum spanning tree (Prim) | Section 5 |
| `11_clrs_lcs_network_brief.txt` | CLRS-Text longest common subsequence | Section 5 |
| `11_satlib_network_brief.txt` | SATLIB random 3-SAT | Section 5 |
| `11_planbench_network_brief.txt` | PlanBench Blocksworld, Mystery Blocksworld and Logistics | Section 5 |

Sudoku-Extreme, GraphWalks and every later build (`04b`, `11`, `13`) were written by headless Claude Code sessions whose permissions blocked the web and anything outside the build folder. `13b_maze_size_limit_rule_answer.txt` is the one fixed answer one of the weight-cap builders received to a question about the rules. The other transformers were written by Claude Opus 5.5 subagents under the rules in their briefs, which independent audits checked afterwards.

## Claude solving mazes directly (Figure 10)

| File | Bar of Figure 10 |
|---|---|
| `08_maze_claude_alone_run1_paper.txt` | The paper's prompt, which the other four change. Figure 10's first bar is the paper's own published result. |
| `08_maze_claude_alone_run2_bfs.txt` | Told how to search |
| `08_maze_claude_alone_run3_one_cell_per_token.txt` | Told how, one cell per token |
| `08_maze_claude_alone_run5_path_only.txt` | Told how, answer with just the path |
| `08_maze_claude_alone_run4_no_grid.txt` | Told how, just the path, the maze as a list of cells instead of a grid |
| `10_maze_claude_code_check.txt` | Told how to search, answered inside Claude Code with no tools instead of through the API (Section 3) |

Each file gives the settings, the system prompt and the template of the user message. The maze itself goes where the template says.

## The whole maze transformer in one answer (Section 3)

| File | What it is |
|---|---|
| `09_maze_whole_network_one_answer.txt` | Claude writes the whole maze transformer in a single answer, with no tools and no way to test it. |
| `09b_maze_whole_network_one_answer_no_loop_count.txt` | The same prompt, with the one sentence that named our transformer's loop count replaced. |

## Claude on its own on the three new benchmarks (Table 9)

| File | What it is |
|---|---|
| `12_new_benchmarks_claude_alone.txt` | Settings, the appended system prompt and the answer instructions. Each item's prompt is the one the written transformer received. |

## Reading the maze transformer from its weights (Section 6)

| File | What it is |
|---|---|
| `14_blind_reading_prompts.txt` | The message each of three fresh sessions received with the weights, the architecture description it was given, and the scorer's prompt. |

The training-free fix for Ouro (Appendix C) came out of a conversation, not a single prompt, so it has no file here.
