# Hand-written looped transformers: weights and runner

This folder holds ten small looped transformers whose weights were written by formulas. None of them was trained.
Each network is one shared block (softmax attention, then a ReLU MLP) that runs R times. The tasks are:
Tower of Hanoi and Checker Jumping (the puzzles from "The Illusion of Thinking"), BIG-Bench Hard (BBH)
boolean_expressions and dyck_languages, BIG-Bench Extra Hard (BBEH) web_of_lies and shuffled_objects, the parity of a bit string, the value at the end of a chain
of assignments, the shortest-path length in a directed graph, and Maze-Hard (the shortest path through a maze).

## Files

- `networks/<name>.npz`: the weights. For every tensor `T` it stores `T.shape`, `T.idx` (flat row-major
  indices) and `T.val` (float64 values) of the entries that are not +0.0 (a few are -0.0). All other entries are 0.
- `networks/<name>.json`: the config: heads per layer (d_k, d_v), mask, readout, loop-count rule R,
  embedding fields, end id, limits, and the vocabulary tables for input and output.
- `run.py`: one standalone script. It needs only Python 3 and numpy.

## Run

```
python run.py hanoi 10
python run.py checker 5
python run.py boolean_expressions "( ( False ) ) or ( True ) is"
python run.py dyck_languages "Complete the rest of the sequence, making sure that the parentheses are closed properly. Input: [ { [ ( ) ] }"
python run.py web_of_lies question.txt
python run.py shuffled_objects question.txt
python run.py parity 0110101
python run.py chains "x3 = 7; x5 = x3 + 4; x9 = x5 - 2; ? x9"
python run.py graph "0>1 1>2 2>3 5>0 ? 0 3"
python run.py graph --random 30
python run.py maze maze.txt
python run.py maze --random 31
```

- The puzzle commands print the moves, then a check by a small simulator: every move is legal, the goal is
  reached, and the number of moves is optimal (2^N - 1 for Hanoi, (N+1)^2 - 1 for Checker Jumping).
- BBH inputs use the official BBH format, so boolean_expressions items end in " is".
- For BBEH, give a file or the question text itself. A final newline in the file is dropped.
- parity: a string of 0s and 1s (up to 4,095 bits). It prints 0 or 1.
- chains: statements separated by `;`: `xA = c`, `xA = xB + c` or `xA = xB - c`, then `? xA`. Variable ids
  are 0..63 and constants 0..99; every value must stay in 0..99. It prints the value of the queried variable.
- graph: `u>v` for an edge from u to v, a bare id for a node with no edges, then `? A B`. Node ids are 0..63.
  It prints the shortest-path length from A to B, or `unreachable`.
- maze: a file with one row per line, using `#` (wall), space (open), `S` and `G`, as in the Maze-Hard dataset
  (one line of side x side characters also works). It prints the grid with the path marked `o`, then a check by
  BFS: the path is valid, it is a shortest path, and it is the one that is smallest in move order
  Down < Left < Right < Up (the dataset's tie-break). The same weights serve any square grid with side up to 128,
  because the position tables are written for 128 x 128 and only R grows. Grids larger than 100 x 100 were
  checked on only a few mazes.
- For parity, chains, graph and maze, `--random n` makes an input with n bits, n hops, n nodes or side n, prints
  it and runs it. Parity, chains and graph also print a check: counting the ones, following the links, or a BFS.
- Run time grows fast with N for Hanoi, since the answer has 2^N - 1 moves. On a laptop, N = 10 takes about
  a second and N = 14 about ten seconds; each extra disk makes it three to four times slower. A BBEH question
  takes 5 to 10 seconds, parity with 300 bits about 3 seconds, and a 30 x 30 maze (R = 1800) about 30 seconds.

## What run.py does

Every network runs the same loop. Only its config and weights differ.

1. Embed: each token is a short list of field values, and h is the sum of one row from each field's table.
   Some networks also have a position field, whose value is the token's position.
2. Run the shared block R times. Each layer adds softmax attention (scores q.k / sqrt(d_k), under the
   network's mask), then a ReLU MLP.
3. Read out: logits = h W_out + b_out at the last token (maze: at every cell at once).
4. Decode greedily: append the argmax id and stop at the end id. Parity, chains and graph give one answer id.

The masks are:

- causal: Hanoi, Checker and parity.
- none: BBH, chains, graph and maze. For BBH the whole sequence runs again from scratch at each step.
- prefix: BBEH. The input tokens see the whole input. Each generated token sees the input, the earlier
  generated tokens and itself.

Under the causal and prefix masks, keys and values are cached. This is exact, because earlier rows never see
later ones. Only the input and output code of run.py is task-specific: turning the prompt into token ids, turning
the ids into the answer text, the answer checks, and the `--random` input makers.

| network | d_model | layers | heads | nonzero weights | mask | R |
|---|---|---|---|---|---|---|
| hanoi | 264 | 1 | 1 | 3,686 | causal | 2 |
| checker | 1,367 | 1 | 1 | 11,062 | causal | 3 |
| boolean_expressions | 73 | 2 | 6 | 6,756 | none | n (tokens in the call) |
| dyck_languages | 24 | 2 | 2 | 8,491 | none | 12 |
| web_of_lies | 462 | 27 | 25 | 44,550 | prefix | 14 + ceil(n_input / 20) |
| shuffled_objects | 320 | 9 | 13 | 40,316 | prefix | 2 + ceil(log2 n_input) |
| parity | 6 | 1 | 1 | 8,227 | causal | n (bits) |
| chains | 10 | 1 | 1 | 698 | none | n (tokens: statements + query) |
| graph | 21 | 3 | 5 | 642 | none | n (tokens: nodes + edges + query) |
| maze | 35 | 1 | 4 | 922 | none | 2n (n = cells) |

Use float64 and the exact softmax `exp(s - max) / sum` from run.py: attention scores reach about 1e13, and a library softmax (torch.softmax) returned all-zero rows, so web_of_lies scored 139/180 instead of 180/180.
