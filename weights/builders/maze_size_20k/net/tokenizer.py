"""Generic character-level tokenizer.

Every character of the text is one token with three fields:
  0: the character's code point, clamped to 127 (all non-ASCII characters share code 127);
  1: its line index (number of '\\n' characters before it), clamped to 31;
  2: its column index (characters since the last '\\n' before it), clamped to 31.
A newline character belongs to the line it ends. Works on any string.
"""
import numpy as np


def tokenize(text: str):
    out = np.zeros((len(text), 3), dtype=np.int64)
    line = col = 0
    for i, ch in enumerate(text):
        out[i] = (min(ord(ch), 127), min(line, 31), min(col, 31))
        if ch == "\n":
            line, col = line + 1, 0
        else:
            col += 1
    return out
