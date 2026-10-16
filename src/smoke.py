
CONDITIONS = ("easy", "multi_hop", "unanswerable")


def select_smoke_examples(examples):
    """First example of each condition, in benchmark order."""
    selected = []

    for condition in CONDITIONS:
        for example in examples:
            if example["condition"] == condition:
                selected.append(example)
                break
        else:
            raise ValueError(f"No example with condition {condition!r}.")

    return selected
