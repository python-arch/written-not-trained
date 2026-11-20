"""Generic number-aware character tokenizer.

Splits any text into tokens: a decimal number literal (digits, optionally '.' and more digits) is one token;
every other character is its own token. Each token is a tuple of 4 fields:
  0 cls  : character code (clipped to 255) for single-character tokens, 256 for a number token
  1 ival : integer part of a number token (clipped to 999), else 0
  2 fval : first three fractional digits of a number token as an integer 0..999 (right-padded), else 0
  3 pos  : token position (clipped to 8191)
Nothing here knows about any task.
"""
import re

import numpy as np

_RX = re.compile(r"[0-9]+(?:\.[0-9]+)?|.", re.S)
NUM = 256


def tokenize(text):
    out = []
    for i, m in enumerate(_RX.finditer(text)):
        t = m.group(0)
        p = min(i, 8191)
        if t[0] in "0123456789":
            ip, _, fp = t.partition(".")
            out.append((NUM, min(int(ip), 999), int((fp + "000")[:3]), p))
        else:
            out.append((min(ord(t), 255), 0, 0, p))
    return np.array(out, dtype=np.int64).reshape(-1, 4)
