"""PlanBench scoring for this build (task 1, plan generation): the plan is checked by replaying it. The response's
text up to "[PLAN END]" (all of it if absent), with lines starting with three backticks dropped, is read one line per
action, in the domain's wording as in the prompt's example plan (for example "unstack the red block from on top of
the blue block"); case, surrounding spaces and a final period are ignored, blank lines are skipped. Starting from the
instance's initial state, every action's preconditions must hold when it is taken (STRIPS: add and delete effects),
and the goal must hold at the end. A line that is not an action of the domain makes the plan invalid. The grader
uses tools/domains.json (the domains' actions as in PlanBench's PDDL files, and their wording)."""
import json, os, re

DOMAINS = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "domains.json")))


def parse(text, item):
    """The plan, one list per line of the (action, [objects]) readings that line matches (Logistics words loading a
    truck and an airplane the same way), or None if a line is not an action of the domain."""
    if text is None:
        return None
    text = text.split("[PLAN END]")[0]
    lines = [l.strip().rstrip(".").strip().lower() for l in text.split("\n") if not l.lstrip().startswith("```")]
    nl2obj = {v.lower(): k for k, v in item["objects_nl"].items()}
    names = "|".join(sorted(map(re.escape, nl2obj), key=len, reverse=True))
    pats = [(a, re.compile("^" + re.escape(t).replace(r"\{\}", f"({names})") + "$"))
            for a, t in DOMAINS[item["domain"]]["nl"].items()]
    plan = []
    for l in lines:
        if not l:
            continue
        reads = [(a, [nl2obj[g] for g in m.groups()]) for a, p in pats for m in [p.match(l)] if m]
        if not reads:
            return None
        plan.append(reads)
    return plan


def verdict(text, item):
    """'solved', 'unreadable' (a line that is not an action), 'invalid' (an action whose preconditions fail) or
    'goal not reached'."""
    plan = parse(text, item)
    if plan is None:
        return "unreadable"
    acts = DOMAINS[item["domain"]]["actions"]
    state = {tuple(a) for a in item["init"]}
    for reads in plan:
        for a, args in reads:                     # the first reading whose preconditions hold is the action taken
            s = acts[a]
            if len(args) != len(s["params"]):
                continue
            b = dict(zip(s["params"], args))
            g = lambda atom, b=b: tuple([atom[0]] + [b.get(x, x) for x in atom[1:]])
            if all(g(p) in state for p in s["pre"]):
                state = (state - {g(d) for d in s["del"]}) | {g(x) for x in s["add"]}
                break
        else:
            return "invalid"
    return "solved" if all(tuple(x) in state for x in item["goal"]) else "goal not reached"


def score(response: str, item: dict) -> float:
    """The DEV checker's interface: one problem's score (1 solved, 0 not)."""
    return float(verdict(response, item) == "solved")
