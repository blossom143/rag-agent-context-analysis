import json
import random
from pathlib import Path

from data import load_hotpotqa, get_context


SEED = 42
N_PER_CONDITION = 20


def get_supporting_titles(example):
    return set(example["supporting_facts"]["title"])


def build_easy_example(example):
    documents = get_context(example)
    supporting_titles = get_supporting_titles(example)

    supporting_docs = [
        doc for doc in documents
        if doc["title"] in supporting_titles
    ]

    if len(supporting_docs) != 2:
        return None

    return {
        "id": example["id"],
        "question": example["question"],
        "answer": example["answer"],
        "condition": "easy",
        "documents": documents,
        "initial_documents": documents,
        "supporting_titles": list(supporting_titles),
        "gold_action": "ANSWER",
    }


def build_multihop_example(example):
    documents = get_context(example)
    supporting_titles = get_supporting_titles(example)

    supporting_docs = [
        doc for doc in documents
        if doc["title"] in supporting_titles
    ]

    if len(supporting_docs) != 2:
        return None

    first_support = supporting_docs[0]
    second_support = supporting_docs[1]

    initial_documents = [
        doc for doc in documents
        if doc["title"] != second_support["title"]
    ]

    return {
        "id": example["id"],
        "question": example["question"],
        "answer": example["answer"],
        "condition": "multi_hop",
        "documents": documents,
        "initial_documents": initial_documents,
        "supporting_titles": list(supporting_titles),
        "search_reveal": second_support,
        "gold_action": "SEARCH",
    }


def build_unanswerable_example(example):
    documents = get_context(example)
    supporting_titles = get_supporting_titles(example)

    remaining_documents = [
        doc for doc in documents
        if doc["title"] not in supporting_titles
    ]

    if len(remaining_documents) < 3:
        return None

    return {
        "id": example["id"],
        "question": example["question"],
        "answer": example["answer"],
        "condition": "unanswerable",
        "documents": remaining_documents,
        "initial_documents": remaining_documents,
        "supporting_titles": list(supporting_titles),
        "gold_action": "ABSTAIN",
    }


def main():
    random.seed(SEED)

    dataset = load_hotpotqa("validation")

    candidates = {
        "easy": [],
        "multi_hop": [],
        "unanswerable": [],
    }

    # Build candidates first.
    for example in dataset:
        easy = build_easy_example(example)
        if easy is not None:
            candidates["easy"].append(easy)

        multi = build_multihop_example(example)
        if multi is not None:
            candidates["multi_hop"].append(multi)

        unanswerable = build_unanswerable_example(example)
        if unanswerable is not None:
            candidates["unanswerable"].append(unanswerable)

    # Randomize candidates so we don't systematically select
    # only the first questions in the validation set.
    for condition in candidates:
        random.shuffle(candidates[condition])

    # Select examples while enforcing UNIQUE question IDs.
    selected = []
    used_ids = set()

    for condition in ["easy", "multi_hop", "unanswerable"]:
        count = 0

        for example in candidates[condition]:
            if example["id"] in used_ids:
                continue

            selected.append(example)
            used_ids.add(example["id"])
            count += 1

            if count == N_PER_CONDITION:
                break

        if count < N_PER_CONDITION:
            raise RuntimeError(
                f"Could only find {count} unique examples "
                f"for {condition}, expected {N_PER_CONDITION}."
            )

    random.shuffle(selected)

    # Final validation.
    ids = [example["id"] for example in selected]

    assert len(selected) == 60
    assert len(set(ids)) == 60

    condition_counts = {}

    for example in selected:
        condition = example["condition"]
        condition_counts[condition] = condition_counts.get(condition, 0) + 1

    assert condition_counts == {
        "easy": 20,
        "multi_hop": 20,
        "unanswerable": 20,
    }

    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / "benchmark_60.json"

    with open(output_file, "w") as f:
        json.dump(selected, f, indent=2)

    print("Benchmark created.")
    print("Easy:", condition_counts["easy"])
    print("Multi-hop:", condition_counts["multi_hop"])
    print("Unanswerable:", condition_counts["unanswerable"])
    print("Total:", len(selected))
    print("Unique IDs:", len(set(ids)))
    print("Saved to:", output_file)


if __name__ == "__main__":
    main()