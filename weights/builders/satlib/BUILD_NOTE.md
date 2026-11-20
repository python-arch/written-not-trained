# BUILD NOTE: looped transformer that decides uniform random 3-SAT

Files: `work/net/{network.json, weights.npz, tokenizer.py}`. `python3 -I work/write_weights.py` writes the network
files (bit-identical on reruns). Helpers I used for my own testing: `work/sim.py` (step simulator), `work/gen.py`
(random 3-SAT and prompt builder), `work/fastrun.py`, `work/test_own.py`, `work/batch_own.py`.

## What the network computes
DPLL with chronological backtracking, **parallel unit propagation** and **Jeroslow-Wang (two-sided) branching**.
**One loop of the block is one DPLL step.** The search is complete, so `UNSAT` is only written after the search has
refuted every branch.

- **Tokenizer** (generic): a single leading space or newline is folded into the next token. Signed integers become
  value tokens (≤3 digits per chunk), and every other character is its own token. The fields are kind (whitespace ×
  char), signed number, position, line index and column. A 50-variable prompt becomes 1224 tokens (4 per clause line).
- **Memory.** Token positions 1..50 (inside the instruction line) are *cells*. Cell v holds variable v (T, F, level,
  is-decision) and decision level v (flipped flag, decision variable). curL, SAT-done and UNSAT-done are held at
  every position.
- **Block** (5 layers, 5 heads, d_model 188):
  1. Each literal token looks up its variable's cell (one-hot match), giving literal true/false. The binary state
     is re-snapped.
  2. A same-line-mean head evaluates every clause line (#true and #false literals). Per literal token: *eligible* (an
     open literal of an open clause) and *forced* (the open literal of a unit clause). Per clause: *conflict* and
     *active*.
  3. A per-cell head gathers the cell's literal tokens with a fixed sink at position 0. It gives forced-positive,
     forced-negative and the JW score S/(S+4) with its sign preference. A global priority head gives conflict? else
     forced? else open clause? else the header's V.
  4. A selector head whose query carries the conflict flag. On a conflict it picks the highest unflipped level and
     its decision variable. Otherwise it picks the unassigned cell with the largest JW score (ties go to the lowest
     index). Then the mode is chosen: backtrack / UNSAT / propagate / SAT / decide / halted.
  5. The state update (MLP only).
- A variable forced both ways is set T=F=1. That makes both its literals false, so the next step sees a conflict
  clause.
- **Readout**: position 0 writes `SAT` / `UNSAT` / `UNKNOWN`. When SAT, cell k (k ≤ V) writes ` k` or ` -k`. Unassigned
  variables are written positive.
- All flags pass through margin-robust ReLU snaps, so the state stays at exact integers. Attention gaps are ≥30
  (mostly ≥40), so dominant keys get weight 1 to float32 precision.

## Size
- d_model 188; d_ff per layer [44, 28, 48, 58, 35]; heads: L1 1 (dk 50), L2 1 (dk 20), L3 2 (dk 51, 1), L4 1 (dk 2).
- Parameters: **1,649,641 total, 27,228 nonzero**. The embedding tables account for most of the total: position 4096
  rows, number 2001 rows.

## R rule and compute
- R = n (the number of tokens): `{"const": 0, "lin": 1, "div": 1, "log2": 0}`.
  - uf20-91 prompts: n = R = 716, **2.52e11 FLOPs**.
  - uf50 / uuf50 prompts: n = R = 1224, **9.71e11 FLOPs** (about 1% of the 1e14 cap).

## Loops needed
Steps the search needs (simulator, which matches the network exactly; see below), on the DEV formulas:

| set | mean | median | max | R |
|---|---|---|---|---|
| uf20-91 | 25 | 19 | 58 | 716 |
| uf50-218 | 106 | 68.5 | 430 | 1224 |
| uuf50-218 | 282 | 279 | 520 | 1224 |

On 12,000 of my own random 50-variable/218-clause formulas:

| | count | mean | 99th pct | 99.9th pct | max |
|---|---|---|---|---|---|
| UNSAT | 6070 | 285 | 527 | 627 | 760 |
| SAT | 5930 | 114 | 401 | 507 | 672 |

None needed more than 800 steps, well under R = 1224. A formula that needs more loops than R gets `UNKNOWN`
(position 0 writes it when neither halt flag is set). Once halted, the state is frozen, and the answer stays put for
the remaining loops.

## Verification on my own formulas (not DEV)
- 520 random formulas (320 with V=50/C=218, both SAT and UNSAT; 200 with V=20/C=91), run with the runner's float32
  arithmetic. Every one was solved, and the network's loop count equalled the simulator's step count on every
  formula.
- 16 more in both float64 and float32, also all solved with matching counts.
- 3 own prompts through `tools/dev_run.py --own` with full R: correct.

## DEV results and runs used
All 300 DEV problems were run on the backend `numpy32` with the same network files (weights.npz sha 690ec7770423):

| set | solved | wrong | no answer |
|---|---|---|---|
| uf20-91 | **100 / 100** | 0 | 0 |
| uf50-218 | **100 / 100** | 0 | 0 |
| uuf50-218 | **100 / 100** | 0 | 0 |

**DEV runs used: 4 of 20** (16 left).
- **Unrecorded run (all 300 ids).** Stopped by the 2-hour limit on background commands after id 153, so it wrote no
  ledger line. The orchestrator confirmed it counts as one run. Its log is `work/dev_run1.log`. It gave the results
  for ids 0–153.
- **Ledger run 1** (UNSAT ids 200–249). At that time `tools/dev_run.py` did not score SATLIB items (a fault in the
  frame, since fixed), so its ledger scores are null. I graded its printed responses with the unmodified
  `tools/grader.py` (`work/grade_logs.py`): 50/50 solved. Its log is `work/dev_runA.log`.
- **Ledger run 2** (uf50 ids 154–199): 46/46 solved, scored by the fixed tool. Its log is `work/dev_runB.log`.
- **Ledger run 3** (UNSAT ids 250–299): 50/50 solved, scored by the fixed tool. Its log is `work/dev_runC.log`.

Nothing about the network was changed or tuned after any DEV run. The design and weights were fixed before the first
one.

## Worries
- I can't run torch32 here. Margins are designed for float32: the JW argmax key has a tie-break step of 200 against
  float32 rounding of about 2 at a magnitude of 1e7, and every flag is snapped with ≥0.25 margins. numpy32 and numpy
  agree with the simulator on everything I've run.
- The network relies on the prompt layout: the instruction is one line of at least 51 tokens (the cells), the header
  is on line 1 with V as its first number, and every clause line has 4 tokens (3 literals + `0`). This is the format
  of all three sets.
- V ≤ 50 and variable indices ≤ 50 are built in (VMAX = 50).
