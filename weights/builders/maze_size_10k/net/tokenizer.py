"""Generic character tokenizer. One token per character (every character of the text, unchanged), with fields:
  0..6  bit k of the character's code point (k = 0..6)
  7     1 if the code point is 128 or more, else 0
  8     line number of the character (newlines before it), mod 64
  9     column of the character (characters since the last newline), mod 64
Nothing here depends on what the text is about."""
import numpy as np

MOD = 64


def tokenize(text: str):
    out = []
    line = col = 0
    for ch in text:
        o = ord(ch)
        out.append([(o >> k) & 1 for k in range(7)] + [1 if o >= 128 else 0, line % MOD, col % MOD])
        if ch == "\n":
            line, col = line + 1, 0
        else:
            col += 1
    return np.asarray(out, dtype=np.int64).reshape(len(out), 10)
