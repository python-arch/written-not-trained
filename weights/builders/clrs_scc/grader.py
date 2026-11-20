"""CLRS-Text scoring for this build: exact match of the final output. The dataset's answers are the execution
trace, then '|', then the final output. The final output of a response is the text after its last '|' (the whole
text if it has none), with lines starting with three backticks dropped and whitespace stripped at both ends; it
must equal the problem's final output exactly. Anything else scores 0."""


def final(text):
    if text is None:
        return None
    text = "\n".join(line for line in text.split("\n") if not line.lstrip().startswith("```"))
    return text.split("|")[-1].strip()


def solved(text, answer) -> bool:
    return final(text) == answer.strip()


def score(response: str, item: dict) -> float:
    """The DEV checker's interface: one problem's score (1 solved, 0 not)."""
    return float(solved(response, item["answer"]))
