"""Sudoku-Extreme scoring, copied from lcrh/lattice-deduction-transformers baselines/frontier_llm/run.py (commit
f0c582a75948): `lines` verbatim, and the Sudoku branch of `score`. A response is solved if its lines, each stripped,
joined, equal the solution's 81 digits."""


def lines(text):
    text = text.strip("\n")
    if text.startswith("```"):
        text = "\n".join(line for line in text.split("\n") if not line.startswith("```"))
    return text.split("\n")


def solved(text, answer) -> bool:
    if text is None:
        return False
    return "".join(line.strip() for line in lines(text)) == answer


def score(response: str, item: dict) -> float:
    """The DEV checker's interface: one problem's score (1 solved, 0 not)."""
    return float(solved(response, item["answer"]))
