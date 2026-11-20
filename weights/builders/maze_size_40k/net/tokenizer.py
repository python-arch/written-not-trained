"""Generic character tokenizer.

One token per character of the text, exactly as given. Each token has five fields:
  0: low nibble of the code point   (ord(c) % 16)
  1: next nibble of the code point  ((ord(c) // 16) % 16)
  2: position, base-32 digit 0      (i % 32)
  3: position, base-32 digit 1      ((i // 32) % 32)
  4: 1 for the first token, else 0
Works on any text (code points and positions wrap around); it knows nothing about any task.
"""


def tokenize(text: str):
    return [(ord(c) % 16, (ord(c) // 16) % 16, i % 32, (i // 32) % 32, int(i == 0)) for i, c in enumerate(text)]
