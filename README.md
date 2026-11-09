# Written, Not Trained

**Claude hand-wrote a tiny transformer that beats Claude at mazes.**

Abdelrahman Elsayed\*, Ahmed Heakl\*, Salman Khan · MBZUAI

**[Read the post](https://python-arch.github.io/written-not-trained/)** · Run it in your browser: [mazes](https://python-arch.github.io/written-not-trained/demo/maze.html) · [Tower of Hanoi](https://python-arch.github.io/written-not-trained/demo/index.html)

## What this is

Claude Opus 5.5 designed these neural networks and set every weight itself, with nothing learned. On Maze-Hard, Claude solves 13 of 50 test mazes on its own. The tiny transformer it wrote solves all 1,000. Its single layer runs again and again, until the whole path is found.

## Results

| Task | Hand-written network | Compared with |
|---|---|---|
| **Maze-Hard** (30×30 mazes) | **All 1,000 test mazes**, with 19,936 weights | Claude Opus 5.5: 13 of 50 test mazes (Davis et al., NeurIPS 2026) |
| **Shortest routes**, longer than any it was checked on | **100%**, with 6,966 weights | **0%** for trained models up to 99× bigger |
| **Parity** (odd or even number of 1s) | **Right at every length tested, up to 4,095 bits**, checked only up to 20 | Trained models: chance beyond 21 bits |
| **Reasoning chains** (`x1 = x0 + 3`, …) | **100% at 12 to 32 steps**, checked only up to 8 | Best trained model: 93.7% on its training lengths, 3.5% beyond |
| **BIG-Bench Hard:** Boolean Expressions / Dyck Languages | **100% / 100%** | Best of four small open models: 69.6% / 33.2% |
| **Tower of Hanoi, 20 disks** | **1,048,575 moves, zero mistakes** | MAKER (2025): over 3 million LLM calls |
| **Checker Jumping, up to 40 per side** | **Every size, fewest possible moves** | Claude 3.7 Sonnet and DeepSeek-R1 (2025): failed from about 3 per side |
| **BIG-Bench Extra Hard:** Web of Lies / Shuffled Objects | **100% / 100%** | Best LLM we list: 81.2 / 56.9 (Oct 2025) |
| **Ouro-1.4B**, a looped language model trained by others, 8 loops on GSM8K (grade-school math) | **75.2%** with a fix that needs no training | 66.6% without it |

## Run it in Python

`weights/` holds all ten networks (Maze-Hard, graph search, parity, reasoning chains, Boolean Expressions, Dyck Languages, Tower of Hanoi, Checker Jumping, Web of Lies and Shuffled Objects) and `run.py`, one short loop that runs any of them. It needs only numpy.

```
cd weights
python run.py maze --random 30
python run.py hanoi 10
```

Each command prints the answer and checks it. `weights/README.md` covers all ten.

## Credits

This work builds on [Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/) (Takehi, Teoh, Zhang and Khattab, MIT CSAIL, 2026). The benchmarks are Maze-Hard (Wang et al.), BIG-Bench Hard (Suzgun et al.) and BIG-Bench Extra Hard (Kazemi et al.), and the puzzles come from Shojaee et al. (2025). The post lists all related work.

## Citation

If you use this work, please cite it:

```bibtex
@misc{elsayed2026written,
  title        = {{Written, Not Trained: Claude hand-wrote a tiny transformer that beats Claude at mazes}},
  author       = {Elsayed, Abdelrahman and Heakl, Ahmed and Khan, Salman},
  year         = {2026},
  month        = nov,
  howpublished = {\url{https://python-arch.github.io/written-not-trained/}},
  note         = {Blog post}
}
```

## License

The code and weights are under the MIT license ([LICENSE](LICENSE)). The text, figures and animations are under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Anyone can use, copy and change all of it, with credit to the authors. It comes as is, with no warranty.
