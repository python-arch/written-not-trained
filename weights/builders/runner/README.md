# Generic looped-transformer runner (harness v1.1)

`looped.py` runs any network in this format; `flops.py` counts its compute per prompt.

## A network folder

- `network.json`
  ```
  {"d_model": 64,
   "fields": ["name of field 0", "name of field 1", ...],
   "layers": [{"heads": [{"dk": 8, "dv": 8, "mask": "none"}, ...], "d_ff": 128}, ...],
   "loops": {"const": 0, "lin": 1, "div": 1, "log2": 0},
   "alphabet": ["", "a", "b", ...]}
  ```
  - `layers` is the one shared block, in order. A layer may have no heads; it then has only its MLP.
  - `mask` is `"none"` or `"causal"`.
  - R = const + lin · ⌈n / div⌉ + log2 · ⌈log₂ max(n, 2)⌉, with n the number of tokens.
  - `alphabet` is a list of strings. The readout writes one entry per position, and `""` writes nothing.
- `weights.npz`, one array per name (float values; stored as float64 or float32):

  | name | shape |
  |---|---|
  | `emb.<f>` (f = 0, 1, …; one per field) | (table size of field f, d_model) |
  | `L<l>.Wq`, `L<l>.Wk` | (d_model, Σ dk over the layer's heads); heads in order |
  | `L<l>.Wv` | (d_model, Σ dv) |
  | `L<l>.Wo` | (Σ dv, d_model) |
  | `L<l>.W1`, `L<l>.b1` | (d_model, d_ff), (d_ff,) |
  | `L<l>.W2`, `L<l>.b2` | (d_ff, d_model), (d_model,) |
  | `out.W`, `out.b` | (d_model, len(alphabet)), (len(alphabet),) |

  An array may instead be stored sparse as three arrays: `<name>.shape`, `<name>.idx` (flat indices into the
  row-major array) and `<name>.val`. Missing entries are 0. Layers without heads omit Wq, Wk, Wv and Wo.
- `tokenizer.py` defines `tokenize(text: str)`. It returns an (n, F) array (or a list of n tuples) of non-negative
  integers, one column per field, each below the size of its embedding table.

## What the runner computes

```
T = tokenize(text)                         text exactly as given
h = Σ_f E_f[T[:, f]]
repeat R times, for each layer:
    h += concat_heads(softmax(q kᵀ / √dk + mask) v) W_o
    h += relu(h W1 + b1) W2 + b2
logits = h W_out + b_out                   every position
response = "".join(alphabet[argmax logits[i]] for i = 0 … n−1)     ties: lowest index
```

The response is returned unchanged: nothing is added before, between or after the written strings.

## Backends

- `numpy`: float64, exact softmax (the maximum is subtracted), attention in chunks of query rows sized so that long
  prompts fit in memory.
- `numpy32`: the same in float32 throughout. It follows the GPU path (`torch32`) as closely as numpy can, so float32
  problems show up on a machine without a GPU. The fused kernel sums in its own order, so the last bits can still
  differ.
- `torch32`: float32 on the GPU with the fused attention kernel, pinned to the memory-efficient kernel. TF32 is off.
  Head dims are zero-padded to a multiple of 8, which changes no value.
- `torch64`: float64, the same exact softmax as numpy.

## Compute

`flops.py` counts dense operations from the shapes, n and R (a multiply-add counts 2; see its docstring). The runner
refuses a prompt whose count exceeds `CAP`, which is set in the build folder's `build.json`.

```
python -I runner/looped.py <network_dir> <text_file> [--backend numpy|numpy32|torch32|torch64]
```
