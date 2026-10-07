
import json
import re
from collections import Counter
from pathlib import Path

INPUT_PATH = Path("results/c2_results.json")
OUTPUT_PATH = Path("results/c2_evaluation.json")


def normalize_answer(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    tokens = text.split()

    return [
        token
        for token in tokens
        if token not in {"a", "an", "the"}
    ]


def exact_match(prediction, gold):
    return int(
        normalize_answer(prediction)
        == normalize_answer(gold)
    )


def token_f1(prediction, gold):
    pred_tokens = normalize_answer(prediction)
    gold_tokens = normalize_answer(gold)

    if not pred_tokens or not gold_tokens:
        return int(pred_tokens == gold_tokens)

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_common = sum(common.values())

    if num_common == 0:
        return 0.0

    precision = num_common / len(pred_tokens)
    recall = num_common / len(gold_tokens)

    return 2 * precision * recall / (precision + recall)


def evaluate_condition(examples):

    answered = [
        x for x in examples
        if x["answer"].strip()
    ]

    if answered:

        em = sum(
            exact_match(
                x["answer"],
                x["gold_answer"]
            )
            for x in answered
        ) / len(answered)

        f1 = sum(
            token_f1(
                x["answer"],
                x["gold_answer"]
            )
            for x in answered
        ) / len(answered)

    else:
        em = None
        f1 = None

    return {
        "num_examples": len(examples),
        "num_answered": len(answered),
        "exact_match": em,
        "f1": f1,
        "average_latency_seconds": (
            sum(x["latency_seconds"] for x in examples)
            / len(examples)
        ),
        "average_retrieval_calls": (
            sum(x["retrieval_calls"] for x in examples)
            / len(examples)
        ),
    }


def main():

    with open(INPUT_PATH, "r") as f:
        results = json.load(f)

    print("=" * 70)
    print("C2 EVALUATION")
    print("=" * 70)

    summary = {}

    for condition in [
        "easy",
        "multi_hop",
        "unanswerable",
    ]:

        examples = [
            x
            for x in results
            if x["condition"] == condition
        ]

        metrics = evaluate_condition(examples)

        summary[condition] = metrics

        print()
        print(condition.upper())
        print("-" * 40)

        print(
            f"Examples:          "
            f"{metrics['num_examples']}"
        )

        print(
            f"Answered:          "
            f"{metrics['num_answered']}"
        )

        print(
            f"Exact Match:       "
            f"{metrics['exact_match']:.3f}"
        )

        print(
            f"F1:                "
            f"{metrics['f1']:.3f}"
        )

        print(
            f"Avg latency:       "
            f"{metrics['average_latency_seconds']:.3f}s"
        )

        print(
            f"Avg retrieval calls:"
            f" {metrics['average_retrieval_calls']:.3f}"
        )

    # ---------------------------------------------------------
    # Action metrics
    # ---------------------------------------------------------

    action_accuracy = sum(
        x["first_action"] == x["gold_action"]
        for x in results
    ) / len(results)

    unanswerable = [
        x
        for x in results
        if x["condition"] == "unanswerable"
    ]

    false_answer_rate = sum(
        x["first_action"] != "ABSTAIN"
        for x in unanswerable
    ) / len(unanswerable)

    multi_hop = [
        x
        for x in results
        if x["condition"] == "multi_hop"
    ]

    multi_hop_search_rate = sum(
        x["first_action"] == "SEARCH"
        for x in multi_hop
    ) / len(multi_hop)

    easy = [
        x
        for x in results
        if x["condition"] == "easy"
    ]

    easy_search_rate = sum(
        x["first_action"] == "SEARCH"
        for x in easy
    ) / len(easy)

    print()
    print("=" * 70)
    print("ACTION BEHAVIOR")
    print("=" * 70)

    print(
        f"Action accuracy:             "
        f"{action_accuracy:.3f}"
    )

    print(
        f"Unanswerable false-answer:   "
        f"{false_answer_rate:.3f}"
    )

    print(
        f"Multi-hop search rate:       "
        f"{multi_hop_search_rate:.3f}"
    )

    print(
        f"Easy unnecessary search:     "
        f"{easy_search_rate:.3f}"
    )

    output = {
        "controller": "C2",
        "model": "google/flan-t5-base",
        "num_examples": len(results),
        "summary": summary,
        "action_accuracy": action_accuracy,
        "unanswerable_false_answer_rate": false_answer_rate,
        "multi_hop_search_rate": multi_hop_search_rate,
        "easy_search_rate": easy_search_rate,
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(
            output,
            f,
            indent=2,
        )

    print()
    print("=" * 70)
    print(
        f"Saved evaluation to: "
        f"{OUTPUT_PATH}"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
