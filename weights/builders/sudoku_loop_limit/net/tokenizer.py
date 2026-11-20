"""Generic character tokenizer: one token per Unicode character.

Field 0: the character's byte value (code point if < 256, otherwise 255).
Field 1: the token's position in the text, clipped to 1023.

It knows nothing about the task; it works unchanged on any text."""

MAX_POS = 1024


def tokenize(text: str):
    return [(min(ord(ch), 255), min(i, MAX_POS - 1)) for i, ch in enumerate(text)]
