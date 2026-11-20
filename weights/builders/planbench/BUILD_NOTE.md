# BUILD NOTE: looped transformers for PlanBench task 1 (plan generation)

All weights are written by `python3 -I work/write_weights.py`. Nothing is trained or fitted. Three networks:
`net_blocksworld`, `net_mystery_blocksworld`, `net_logistics`. They share the builder code.

## Frame use

- **Tokenizer (same file in all three).** One token per UTF-8 byte, with two fields: the byte value (0..255) and the
  position, clipped to 16383. It is generic and knows nothing about the task.
- **Embedding.**
  - The byte table gives a 45-dim character class (26 letters, case-folded, plus an upper-case flag; 10 digits;
    `space , . \n [ ] _`; "other") and a constant 1.
  - The position table gives 14 sinusoid pairs (frequencies 2π/2^(k+1)), the raw position, a first-token flag, and
    "canvas" features (below).
- **Block.** A single layer (all heads in parallel, then one MLP), applied R times. Every residual quantity is
  written with *set* semantics (the MLP adds new − old). Every head writes into a scratch group that the same MLP
  reads and then clears. So the layer is idempotent, and each loop pushes the computation one step further along
  its dependency chain.
- **Head types.**
  - Relative offset heads: sinusoid query rotation that reads position p+k.
  - Causal recency heads: score M·cond + λ·pos.
  - Content-lookup heads: score A·(one-hot · one-hot), with the first token as the "not found" default.
- **Readout ("emit").** A fixed range of positions at the start of the prompt (the canvas) carries learned
  positional features that say which piece of which plan step that position writes. Each such position writes one
  alphabet string: a fixed phrase, an object name (BW), or a digit (Logistics) when its step is active. Every other
  position writes `""`. The alphabet strings are the domain's action wording.

## What each network computes

**Blocksworld and Mystery Blocksworld** (same code; only the character patterns and the wording differ):
1. Character windows give local patterns:
   - BW: colour-word starts and ends, and the `f` of "on top of".
   - Mystery: `craves`.
   - Both: the markers `[PLAN END]`, "initial" and "goal".
2. A fact "X on Y" is anchored at the pattern. X and Y are read at fixed offsets. A fact counts only if it comes
   after `[PLAN END]` (the problem, not the example), and its section (initial vs goal) is taken from the most
   recent "initial"/"goal" marker.
3. These are iterated across loops:
   - the height of every initial fact and of every goal fact (height = 1 + height of the fact below it);
   - **stay(X on Y)**: the goal has X on Y or nothing for X; no other block's goal is on Y; and Y stays or is a
     table block with no goal.
4. The plan:
   - Phase 1: for every block that doesn't stay, from the top level down, unstack it and put it down.
   - Phase 2: for every goal fact whose block doesn't stay, from the bottom level up, pick it up and stack it.
   - The canvas has a slot (phase, level 1..6, block 1..12) for each case, so the order comes from position alone.
5. The plan is valid by construction (the argument is in the code comments). It handles up to 12 blocks (12 colour
   names / letters a..l) and towers of up to 7 blocks.

**Logistics:**
1. Character windows parse each fact at the end of its subject name: "is an airport", "is at",
   "is in the city". The digits are read at fixed offsets.
2. Lookups, one hop per loop:
   - for each package slot p (0..9): its goal place G and start place S;
   - their cities X and Y;
   - airports AX and AY;
   - trucks tX and tY and their start places;
   - one airplane (the last one listed) and its start.
3. A fixed 14-action template per package, each action gated by static conditions:
   - same city or not;
   - S = AX;
   - G = AY;
   - already at the goal.

   The "from" place of a drive or flight is the vehicle's current place. That is the destination of its latest
   earlier *required* move, found by a causal recency lookup over the canvas, or its start place if there is none.
   The move is skipped when "from" equals "to". Because a skipped move doesn't change where the vehicle is, this
   needs no sequential chain. There are no return trips.
4. Valid by construction. It handles up to 10 packages and single-digit indices. It assumes one truck per city, as
   the prompt states.

## Sizes, loops, FLOPs

| net | d_model | heads | d_ff | params total (dense) | nonzero | R rule | FLOPs/prompt on DEV (min / mean / max) |
|---|---|---|---|---|---|---|---|
| blocksworld | 487 | 14 | 860 | 9,470,595 | 482,146 | R = 16 (const) | 1.8e11 / 2.1e11 / 2.7e11 |
| mystery_blocksworld | 450 | 14 | 763 | 8,653,382 | 481,867 | R = 16 (const) | 1.6e11 / 1.8e11 / 2.1e11 |
| logistics | 1327 | 28 | 3106 | 33,613,387 | 494,054 | R = 20 (const) | 2.3e12 / 4.0e12 / 5.5e12 |

- Most of the dense size is the 16384-row position table; it is stored sparse.
- Even a 16384-token prompt stays under the 1e14 cap (5.6e12 BW, 2.1e13 Logistics).
- **Convergence on own problems (smallest R that gives the final answer):**
  - BW: R ≥ 11, including a 7-block tower that must be reversed.
  - Logistics: R ≥ 11–14, on edge cases with 1 to 4 cities, 3 airplanes and 10 packages.
  - Answers stay unchanged for every larger R tested (+3 to +4 loops).
- **Readout margin:** the gap between the best and second-best logit is exactly 0.5 at every position. Answers were
  identical on `numpy` and `numpy32` for every own problem compared.

## DEV results (numpy32)

| domain | valid plans | our actions / reference actions | same length as reference |
|---|---|---|---|
| blocksworld | 100 / 100 | 854 / 764 (×1.12) | 59 / 100 |
| mystery_blocksworld | 100 / 100 | 854 / 764 (×1.12) | 59 / 100 |
| logistics (v3, with return trips; superseded) | 85 / 85 | 2966 / 1952 (×1.52) | 1 / 85 |
| logistics (final, v4) | 85 / 85 | 2431 / 1952 (×1.25) | 19 / 85 |

DEV runs used: 4 of 60.
- Runs 1–3 were BW, Mystery and Logistics v3; they were launched at the same time, so all three ledger lines say
  "run 1".
- Run 4 is the final Logistics network.

All three networks in `work/` are byte-identical to the ones that were run: the hashes of `network.json`,
`weights.npz` and `tokenizer.py` match ledger lines 1 (Mystery), 2 (BW) and 4 (Logistics).

## Things that worried me

- **`tools/dev_run.py` does not score PlanBench items.** It grades only items that have an `answer` or
  `answer_nodes` key, and the DEV items have neither. So the ledger rows have `f1: null`. I graded the responses
  printed by `--show` with `tools/grader.py` (`work/grade_logs.py`), without running the network again. I reported
  this rather than changing the tool.
- **Limits that would fail on TEST if it goes beyond DEV:**
  - Block names outside the 12 standard PlanBench colours (Mystery: letters beyond `l`).
  - Towers taller than 7 blocks.
  - Logistics indices of 10 or more, or more than 10 packages.
  - A city with two trucks.
  - Prompts longer than 16384 bytes.
  DEV has at most 5 blocks, 3 cities and 7 packages.
- No GPU was available, so `torch32` was never run. The margins above (0.5 logit gap, attention gaps of 20 or more
  in the scores) are what I rely on for the fused kernel to agree.
- Plans are valid but not optimal. Logistics carries one package at a time (no shared flights), and BW goes through
  the table where a direct move would do.

## Files

- `write_weights.py`: writes all weights.
- `gen.py`: my own problem generator. It reuses only the prompt boilerplate.
- `test_own.py`, `plan_len.py`, `conv_check.py`, `edge_bw.py`, `edge_lg.py`, `cmp_backends.py`, `margins.py`: checks on own
  problems.
- `grade_logs.py`, `dev_lengths.py`: grade and measure the logged DEV responses.
- `logs/`: outputs.
