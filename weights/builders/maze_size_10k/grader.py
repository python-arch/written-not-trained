"""Maze-Hard scoring for this build: exact match of the full solved grid (as HRM and TRM score it). The response,
with newlines at both ends removed, must equal the solved grid exactly: 30 lines of 30 characters joined by newlines.
Anything else scores 0."""


def solved(text, answer) -> bool:
    return text is not None and text.strip("\n") == answer


def score(response: str, item: dict) -> float:
    """The DEV checker's interface: one maze's score (1 solved, 0 not)."""
    return float(solved(response, item["answer"]))
