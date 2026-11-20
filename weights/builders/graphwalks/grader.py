"""GraphWalks extraction and F1, copied from the openai/graphwalks dataset card (version of 2026-09-14; the only
change is that get_list is a plain function instead of a method)."""
import re


def get_list(response: str) -> tuple[list[str], bool]:
        # get the very last line of the response
        line = response.split("\n")[-1]
        # check if formatted correctly
        if "Final Answer:" not in line:
            return [], True
        list_part = re.search(r"\[.*\]", line)
        if list_part:
            result_list = list_part.group(0).strip("[]").split(",")
            # if the list was empty, then get [] not [""]
            result_list = [item.strip() for item in result_list if item.strip()]
            return result_list, False
        else:
            return [], True


def f1(response: str, answer_nodes) -> float:
    sampled_list, failed_to_parse = get_list(response)
    sampled_set = set(sampled_list)
    truth_set = set(answer_nodes)
    n_golden = len(truth_set)
    n_sampled = len(sampled_set)

    if failed_to_parse:
        recall = 0.0
        precision = 0.0
        f1 = 0.0
    elif n_golden == n_sampled == 0:
        recall = 1.0
        precision = 1.0
        f1 = 1.0
    else:
        n_overlap = len(sampled_set & truth_set)
        recall = n_overlap / n_golden if n_golden > 0 else 0
        precision = n_overlap / n_sampled if n_sampled > 0 else 0
        f1 = 2 * (recall * precision) / (recall + precision) if recall + precision > 0 else 0
    return f1


def score(response: str, item: dict) -> float:
    """The DEV checker's interface: one problem's score."""
    return f1(response, item["answer_nodes"])
