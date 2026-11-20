"""Generic byte tokenizer.

The text is encoded as UTF-8. Each byte becomes one token with 11 integer fields:
  fields 0-7 : the 8 bits of the byte (bit 0 = least significant), each 0 or 1;
  fields 8-10: the byte's position in the text modulo 9, 10 and 11 (a residue-number positional code).
It works unchanged on any text and knows nothing about what the text contains.
"""
import numpy as np

MODULI = (9, 10, 11)
N_FIELDS = 8 + len(MODULI)


def tokenize(text: str):
    data = text.encode("utf-8")
    T = np.zeros((len(data), N_FIELDS), dtype=np.int64)
    for i, b in enumerate(data):
        for k in range(8):
            T[i, k] = (b >> k) & 1
        for f, m in enumerate(MODULI):
            T[i, 8 + f] = i % m
    return T
