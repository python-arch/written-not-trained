"""Generic text tokenizer (knows nothing about any task).

Pre-tokenization, in the style of byte/char tokenizers that fold a leading space into the next token and keep
integers as value tokens:
- A single ' ' or '\n' directly before a non-whitespace character is folded into that token as its "leading
  whitespace" (0 none, 1 space, 2 newline). Any other whitespace character is a token of its own.
- An integer is a run of ASCII digits, optionally preceded by '-', split into chunks of at most 3 digits; the sign
  belongs to the first chunk. Every other character is one token.

Fields per token (all clipped to their table sizes):
  0 kind   = leading_ws * 257 + base   (base: char code for codes < 255, 255 for any other char, 256 for an integer)
  1 number = 0 for non-integers, else 1 + 2 * value + sign       (value 0..999, sign 1 if '-')
  2 position in the token sequence                                (clipped to 4095)
  3 line index (number of newlines before the token)              (clipped to 1023)
  4 column: index of the token within its line                    (clipped to 63)
"""

POS_MAX, LINE_MAX, COL_MAX = 4095, 1023, 63


def _digit(c):
    return "0" <= c <= "9"


def tokenize(text):
    out = []
    n, i, line, col = len(text), 0, 0, 0
    while i < n:
        c, ws = text[i], 0
        if c in (" ", "\n") and i + 1 < n and not text[i + 1].isspace():
            ws = 1 if c == " " else 2
            if c == "\n":
                line, col = line + 1, 0
            i += 1
            c = text[i]
        if _digit(c) or (c == "-" and i + 1 < n and _digit(text[i + 1])):
            sign = 1 if c == "-" else 0
            i += sign
            j = i
            while j < n and j - i < 3 and _digit(text[j]):
                j += 1
            value = int(text[i:j])
            kind, num = ws * 257 + 256, 1 + 2 * value + sign
            i = j
        else:
            kind, num = ws * 257 + min(ord(c), 255), 0
            i += 1
        out.append((kind, num, min(len(out), POS_MAX), min(line, LINE_MAX), min(col, COL_MAX)))
        col += 1
        if c == "\n" and ws == 0:          # a newline that was its own token starts a new line
            line, col = line + 1, 0
    return out if out else _empty()


def _empty():
    import numpy as np
    return np.zeros((0, 5), dtype=np.int64)
