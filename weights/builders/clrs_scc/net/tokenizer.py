"""Generic character tokenizer: one token per character of the text.

Field 0: the character's code point, ord(c), with every code point >= 127 mapped to 127 (table size 128).
Field 1: the character's position in the text, 0-based, clipped at 4095 (table size 4096).
Nothing else; it works on any string."""

CHAR_TABLE = 128
POS_TABLE = 4096


def tokenize(text: str):
    return [(min(ord(c), CHAR_TABLE - 1), min(i, POS_TABLE - 1)) for i, c in enumerate(text)]
