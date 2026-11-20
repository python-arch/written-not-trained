"""Generic byte-level tokenizer: one token per UTF-8 byte of the text.

Fields: (byte value 0..255, absolute position clamped to 0..1023). Works on any text; knows nothing about the task.
"""
NPOS = 1024


def tokenize(text: str):
    return [(b, min(i, NPOS - 1)) for i, b in enumerate(text.encode("utf-8"))]
