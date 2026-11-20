# lcs_length looped transformer — build note

Network folder: `work/net/` (`network.json`, `weights.npz`, `tokenizer.py`), written by `python3 -I work/write_weights.py`.

## What it computes
Tokenizer: one token per character, fields (code point clamped to 255, position clamped to 2047). Nothing about the task is in it.

Block of 4 layers, run R times. Every MLP *overwrites* its slots (it adds target − current) and clears the scratch slots its
attention wrote, so re-running the parsing layers on every loop changes nothing.
- **L0**: a uniform causal head gives the prefix means of `[`, `]` and space. The MLP turns each mean into an exact count
  (B, E, S) by summing steps against thresholds (t−½)/(p+1) taken from the position embedding. A quadratic-score head
  checks whether the previous character is `-`.
- **L1**: a head reads S at the latest `[`, so the column index is c = S − S[latest `[`]. The MLP also computes B², c² and
  integer indicators (B=2, E=1, depth=2, B≥4, B≥3, depth≥1, B=4).
- **L2**: two exact-index heads read key[r] (r = B−4) and key[c] from the key list. The MLP computes match, the valid-cell
  flag (a digit at depth 2 in the trace, not after `-`), the anchor flag and the in-trace flag.
- **L3**: three exact-index heads read the LCS value C of the anchors at (r−1,c), (r,c−1) and (r−1,c−1). The MLP does one
  synchronous DP step, with the CLRS tie rule (up when C_up ≥ C_left) and up/diagonal set to 0 on row 0. It also produces
  one-hot b ∈ {0,1,2}, plus a copy letter for every other character of the prompt's `initial_trace`.

Readout: each character of the prompt's `initial_trace` region writes either itself or, for an answered cell, its b digit.
Everything else writes `""`. The output is therefore the b table in exactly the dataset's format (the `-1` layout matches
the initial trace; I checked this on DEV with plain Python).

## Size
d_model 227, d_ff [434, 124, 19, 69], 8 heads. Parameters: 833,752 total, 334,313 nonzero. Most of them are the
position table (2048 × 227).

## R rule
R = 12 (constant). The deepest DP cell for n=10 is at depth 8, so C and b are final by loop 9.

## FLOPs per prompt (runner count)
n=4 (130 tokens): 9.85e8 · n=5 (162 tokens): 1.23e9 · n=10 (403 tokens): 3.18e9. All are far below the 1e14 cap.

## DEV
1 run used (of 20): all 999 DEV problems on `numpy`, **999/999 solved**. Before that I tested only on 300 problems I
generated myself (not counted). The smallest top-1/top-2 logit gap on those was 0.5.

## Worries
- The design assumes what every DEV problem shows: `string` is sorted, with ⌈n/2⌉ zeros, so the x elements are the first
  rows and the y elements are the last columns. An unsorted `string` would break it.
- Attention scores reach about 1e7. float64 (numpy / torch64) is exact enough. **float32 is not**: `numpy32` solved 96.3%
  of my own problems. TEST runs torch64, which does the same float64 exact softmax as numpy.
- The file has 999 lines, not the 1,000 the brief states.
