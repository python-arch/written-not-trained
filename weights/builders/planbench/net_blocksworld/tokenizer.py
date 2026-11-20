"""Generic byte tokenizer: one token per UTF-8 byte of the text, two fields:
field 0 = the byte value (0..255), field 1 = the token's position, clipped to 16383."""
P = 16384


def tokenize(text: str):
    data = text.encode("utf-8")
    return [(b, min(i, P - 1)) for i, b in enumerate(data)]
