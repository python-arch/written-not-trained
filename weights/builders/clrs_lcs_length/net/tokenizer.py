"""Generic character tokenizer: one token per character, two fields.

field 0: the character's code point, clamped to 255 (codes >= 255 share 255)
field 1: the character's position in the text, clamped to 2047
"""


def tokenize(text: str):
    return [(min(ord(ch), 255), min(i, 2047)) for i, ch in enumerate(text)]
