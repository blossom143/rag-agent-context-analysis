
import argparse
import json
import re
from collections import Counter
from pathlib import Path

INPUT_PATH = Path("results/c0_answers_v2.json")
OUTPUT_PATH = Path("results/c0_evaluation_v2.json")
SMOKE_INPUT_PATH = Path("results/smoke/c0_answers_v2.json")
SMOKE_OUTPUT_PATH = Path("results/smoke/c0_evaluation_v2.json")


def normalize_answer(text):
    """Normalize text for HotpotQA-style EM/F1 evaluation."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    tokens = text.split()

    # Remove common English articles.
    tokens = [
        token for token in tokens
        if token not in {"a", "an", "the"}
    ]

    return tokens


def exact_match(prediction, gold):
    if not prediction.strip():
        return int(not gold.strip())

    return int(
        normalize_answer(prediction)
        == normalize_answer(gold)
    )


def token_f1(prediction, gold):
    if not prediction.strip():
        return int(not gold.strip())

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


def evaluate_group(examples):
    em_scores = [
        exact_match(x["answer"], x["gold_answer"])
        for x in examples
    ]

    f1_scores = [
        token_f1(x["answer"], x["gold_answer"])
        for x in examples
    ]

    latencies = [
        x["latency_seconds"]
        for x in examples
    ]

    output_tokens = [
        x["output_tokens"]
        for x in examples
    ]

    num_answered = sum(
        bool(x["answer"].strip())
        for x in examples
    )

    return {
        "num_examples": len(examples),
        "num_answered": num_answered,
        "answer_coverage": num_answered / len(examples),
        "exact_match": sum(em_scores) / len(em_scores),
        "f1": sum(f1_scores) / len(f1_scores),
        # C0 makes no action decision, so there is no output to parse.
        "invalid_output_rate": None,
        "context_truncated_rate": (
            sum(x["context_truncated"] for x in examples) / len(examples)
        ),
        "average_input_tokens": (
            sum(x["input_tokens"] for x in examples) / len(examples)
        ),
        "average_latency_seconds": sum(latencies) / len(latencies),
        "average_output_tokens": sum(output_tokens) / len(output_tokens),
    }


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    input_path = SMOKE_INPUT_PATH if args.smoke else INPUT_PATH
    output_path = SMOKE_OUTPUT_PATH if args.smoke else OUTPUT_PATH

    with open(input_path, "r") as f:
        results = json.load(f)

    print("=" * 70)
    print("C0 EVALUATION")
    print("=" * 70)

    summary = {}

    for condition in ["easy", "multi_hop", "unanswerable"]:

        examples = [
            x for x in results
            if x["condition"] == condition
        ]

        metrics = evaluate_group(examples)
        summary[condition] = metrics

        print()
        print(condition.upper())
        print("-" * 40)
        print(f"Examples:       {metrics['num_examples']}")
        print(f"Coverage:       {metrics['answer_coverage']:.3f}")
        print(f"Exact Match:    {metrics['exact_match']:.3f}")
        print(f"F1:             {metrics['f1']:.3f}")
        print(f"Context trunc:  {metrics['context_truncated_rate']:.3f}")
        print(f"Avg input tok:  {metrics['average_input_tokens']:.1f}")
        print(
            f"Avg latency:    "
            f"{metrics['average_latency_seconds']:.3f}s"
        )
        print(
            f"Avg output tok: "
            f"{metrics['average_output_tokens']:.1f}"
        )

    # C0 always chooses ANSWER.
    correct_actions = sum(
        x["gold_action"] == "ANSWER"
        for x in results
    )

    action_accuracy = correct_actions / len(results)

    # On unanswerable examples, the correct action is ABSTAIN.
    # C0 always answers, so every such example is a false answer.
    unanswerable = [
        x for x in results
        if x["condition"] == "unanswerable"
    ]

    false_answer_rate = sum(
        x["predicted_action"] == "ANSWER"
        for x in unanswerable
    ) / len(unanswerable)

    overall = evaluate_group(results)

    print()
    print("=" * 70)
    print("OVERALL")
    print("=" * 70)
    print(f"Exact Match:                 {overall['exact_match']:.3f}")
    print(f"F1:                          {overall['f1']:.3f}")
    print(f"Answer coverage:             {overall['answer_coverage']:.3f}")

    print()
    print("=" * 70)
    print("ACTION BEHAVIOR")
    print("=" * 70)
    print(f"Action accuracy:             {action_accuracy:.3f}")
    print("Invalid-output rate:         N/A (no decision step)")
    print(
        f"Unanswerable false-answer:   "
        f"{false_answer_rate:.3f}"
    )

    output = {
        "controller": "C0",
        "model": "google/flan-t5-base",
        "input_path": str(input_path),
        "num_examples": len(results),
        "summary": summary,
        "overall": overall,
        "action_accuracy": action_accuracy,
        "invalid_output_rate": None,
        "unanswerable_false_answer_rate": false_answer_rate,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print()
    print("=" * 70)
    print(f"Saved evaluation to: {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
