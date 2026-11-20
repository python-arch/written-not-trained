"""SATLIB scoring for this build: verified correctness. A response, with lines starting with three backticks dropped
and whitespace stripped, is read as words:
- "UNSAT" (alone) claims the formula is unsatisfiable: right only if it is;
- "SAT" followed by one signed literal per variable (each variable 1..V exactly once) claims an assignment: right only
  if the formula is satisfiable and the assignment satisfies every clause;
- anything else, including "UNKNOWN" or nothing, is no answer (a timeout).
Words are matched without regard to case."""


def parse(text):
    if text is None:
        return None
    text = "\n".join(line for line in text.split("\n") if not line.lstrip().startswith("```"))
    w = text.split()
    if len(w) == 1 and w[0].upper() == "UNSAT":
        return ("UNSAT", None)
    if w and w[0].upper() == "SAT":
        try:
            return ("SAT", [int(x) for x in w[1:]])
        except ValueError:
            return None
    return None


def verdict(text, item):
    """'solved', 'wrong' (a wrong claim or a bad assignment) or 'no answer'."""
    p = parse(text)
    if p is None:
        return "no answer"
    if p[0] == "UNSAT":
        return "solved" if item["label"] == "UNSAT" else "wrong"
    lits, V = p[1], item["n_vars"]
    if item["label"] != "SAT" or sorted(abs(x) for x in lits) != list(range(1, V + 1)):
        return "wrong"
    true = set(lits)
    return "solved" if all(any(l in true for l in cl) for cl in item["clauses"]) else "wrong"


def score(response: str, item: dict) -> float:
    """The DEV checker's interface: one problem's score (1 solved, 0 not)."""
    return float(verdict(response, item) == "solved")
