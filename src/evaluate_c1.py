
import json
import re
from collections import Counter
from pathlib import Path

INPUT_PATH = Path("results/c1_results.json")
OUTPUT_PATH = Path("results/c1_evaluation.json")


def normalize_answer(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    tokens = text.split()

    tokens = [
        token
        for token in tokens
        if token not in {"a", "an", "the"}
    ]

    return tokens


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
    # ABSTAIN examples have no generated answer.
    answerable = [
        x for x in examples
        if x["answer"].strip()
    ]

    if answerable:
        em = sum(
            exact_match(x["answer"], x["gold_answer"])
            for x in answerable
        ) / len(answerable)

        f1 = sum(
            token_f1(x["answer"], x["gold_answer"])
            for x in answerable
        ) / len(answerable)
    else:
        em = None
        f1 = None

    return {
        "num_examples": len(examples),
        "num_answered": len(answerable),
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
    print("C1 EVALUATION")
    print("=" * 70)

    summary = {}

    for condition in [
        "easy",
        "multi_hop",
        "unanswerable",
    ]:

        examples = [
            x for x in results
            if x["condition"] == condition
        ]

        metrics = evaluate_condition(examples)
        summary[condition] = metrics

        print()
        print(condition.upper())
        print("-" * 40)

        print(f"Examples:          {metrics['num_examples']}")
        print(f"Answered:          {metrics['num_answered']}")

        if metrics["exact_match"] is not None:
            print(f"Exact Match:       {metrics['exact_match']:.3f}")
            print(f"F1:                {metrics['f1']:.3f}")
        else:
            print("Exact Match:       N/A")
            print("F1:                N/A")

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

    action_correct = sum(
        x["first_action"] == x["gold_action"]
        for x in results
    )

    action_accuracy = action_correct / len(results)

    # How often did C1 answer when it should have abstained?
    unanswerable = [
        x for x in results
        if x["condition"] == "unanswerable"
    ]

    false_answer_count = sum(
        x["first_action"] != "ABSTAIN"
        for x in unanswerable
    )

    false_answer_rate = (
        false_answer_count / len(unanswerable)
    )

    # How often did C1 search on Multi-hop?
    multi_hop = [
        x for x in results
        if x["condition"] == "multi_hop"
    ]

    multi_hop_search_rate = sum(
        x["first_action"] == "SEARCH"
        for x in multi_hop
    ) / len(multi_hop)

    # How often did C1 unnecessarily search on Easy?
    easy = [
        x for x in results
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

    print(f"Action accuracy:             {action_accuracy:.3f}")
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
        "controller": "C1",
        "model": "google/flan-t5-base",
        "num_examples": len(results),
        "summary": summary,
        "action_accuracy": action_accuracy,
        "unanswerable_false_answer_rate": false_answer_rate,
        "multi_hop_search_rate": multi_hop_search_rate,
        "easy_search_rate": easy_search_rate,
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print()
    print("=" * 70)
    print(f"Saved evaluation to: {OUTPUT_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()
