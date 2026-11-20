"""Generic character tokenizer: one token per character of the text, two fields.

field 0: ord(ch) % 7            (a 7-bucket hash of the character code)
field 1: position class          0 = first character of the text, 1 = odd index, 2 = even index (not first)

Works on any text; it knows nothing about the content.
"""


def tokenize(text: str):
    return [(ord(ch) % 7, 0 if i == 0 else (1 if i % 2 else 2)) for i, ch in enumerate(text)]
