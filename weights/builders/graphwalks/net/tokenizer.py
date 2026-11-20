"""Generic byte-chunk tokenizer with a local byte-context window.

The text is encoded as UTF-8 bytes and split into lines at b"\\n" (the newline itself belongs to no chunk). Each line
is cut into chunks of CHUNK = 4 bytes, left to right; an empty line gives one empty chunk. One token per chunk.

Fields of a token whose chunk starts at byte offset p of the whole byte stream:
  0 .. 83  : the byte at stream position p + o for o = -24 .. 59, as byte + 1 (0 = outside the text)   table 257
  84       : index of the chunk within its line, capped at 127                                           table 128
  85       : 1 for the first token of the text, else 0                                                   table 2
  86       : 1 for the last token of the text, else 0                                                    table 2

Nothing here depends on what the text says; it works on any string.
"""
import numpy as np

CHUNK = 4
LO, HI = -24, 60          # window offsets o in [LO, HI)
NWIN = HI - LO
FIELDS = NWIN + 3
SIZES = [257] * NWIN + [128, 2, 2]


def tokenize(text: str):
    b = np.frombuffer(text.encode("utf-8", "surrogatepass"), dtype=np.uint8)
    L = len(b)
    nl = np.flatnonzero(b == 10)
    starts = np.concatenate([[0], nl + 1])            # line starts
    ends = np.concatenate([nl, [L]])                  # line ends (exclusive)
    lens = ends - starts
    nch = np.maximum(1, -(-lens // CHUNK))            # chunks per line
    tok_line = np.repeat(np.arange(len(starts)), nch)
    first_tok = np.concatenate([[0], np.cumsum(nch)[:-1]])
    cidx = np.arange(len(tok_line)) - first_tok[tok_line]
    pos = starts[tok_line] + CHUNK * cidx            # stream offset of each chunk start
    n = len(pos)
    padded = np.zeros(L + NWIN + 2, dtype=np.int64)
    padded[-LO:-LO + L] = b.astype(np.int64) + 1      # padded[i - LO] = byte i + 1
    T = np.zeros((n, FIELDS), dtype=np.int64)
    for k in range(NWIN):
        T[:, k] = padded[pos + k]                     # offset o = LO + k -> padded index pos + o - LO
    T[:, NWIN] = np.minimum(cidx, 127)
    T[0, NWIN + 1] = 1
    T[n - 1, NWIN + 2] = 1
    return T
