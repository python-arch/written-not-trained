# Prompts

These are the first messages Claude Opus 5.5 received in each experiment in the [blog post](https://python-arch.github.io/written-not-trained/). The task, rules and limits are word for word. Details of our compute setup, such as machine names, accounts, paths and job ids, are removed or made generic. The messages Claude received later in each session are not included. They were fixed messages, such as "continue", the outputs of GPU runs it asked for and rule updates.

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

Sudoku-Extreme and GraphWalks were written by headless Claude Code sessions whose permissions blocked the web and anything outside the build folder. The other transformers were written by Claude Opus 5.5 subagents under the rules in their briefs, which independent audits checked afterwards.

## Claude solving mazes directly (Table 1)

| File | Row of Table 1 |
|---|---|
| `08_maze_claude_alone_run1_paper.txt` | The paper's prompt, which the other four change. Table 1's first row is the paper's own published result. |
| `08_maze_claude_alone_run2_bfs.txt` | Told how to search |
| `08_maze_claude_alone_run3_one_cell_per_token.txt` | Told how, one cell per token |
| `08_maze_claude_alone_run5_path_only.txt` | Told how, answer with just the path |
| `08_maze_claude_alone_run4_no_grid.txt` | Told how, just the path, the maze as a list of cells instead of a grid |

Each file gives the settings, the system prompt and the template of the user message. The maze itself goes where the template says.

## The whole maze transformer in one answer (Section 3)

| File | What it is |
|---|---|
| `09_maze_whole_network_one_answer.txt` | Claude writes the whole maze transformer in a single answer, with no tools and no way to test it. |
| `09b_maze_whole_network_one_answer_no_loop_count.txt` | The same prompt, with the one sentence that named our transformer's loop count replaced. |

The training-free fix for Ouro (Appendix C) came out of a conversation, not a single prompt, so it has no file here.
