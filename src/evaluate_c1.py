
import argparse
import json
import re
from collections import Counter
from pathlib import Path

INPUT_PATH = Path("results/c1_results_v2.json")
OUTPUT_PATH = Path("results/c1_evaluation_v2.json")
SMOKE_INPUT_PATH = Path("results/smoke/c1_results_v2.json")
SMOKE_OUTPUT_PATH = Path("results/smoke/c1_evaluation_v2.json")


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


def evaluate_condition(examples):
    # EM/F1 are averaged over every example: an abstention (empty
    # answer) scores 0 whenever the gold answer is nonempty.
    num_examples = len(examples)

    answered = [
        x for x in examples
        if x["answer"].strip()
    ]

    em = sum(
        exact_match(x["answer"], x["gold_answer"])
        for x in examples
    ) / num_examples

    f1 = sum(
        token_f1(x["answer"], x["gold_answer"])
        for x in examples
    ) / num_examples

    # Action metrics use the parsed first decision, never the
    # execution fallback, so INVALID always counts as incorrect.
    action_accuracy = sum(
        x["parsed_action"] == x["gold_action"]
        for x in examples
    ) / num_examples

    invalid_output_rate = sum(
        x["parsed_action"] == "INVALID"
        for x in examples
    ) / num_examples

    decision_truncated_rate = sum(
        x["decision_prompt"]["context_truncated"]
        for x in examples
    ) / num_examples

    return {
        "num_examples": num_examples,
        "num_answered": len(answered),
        "answer_coverage": len(answered) / num_examples,
        "exact_match": em,
        "f1": f1,
        "action_accuracy": action_accuracy,
        "invalid_output_rate": invalid_output_rate,
        "decision_context_truncated_rate": decision_truncated_rate,
        "average_decision_input_tokens": (
            sum(x["decision_prompt"]["input_tokens"] for x in examples)
            / num_examples
        ),
        "average_latency_seconds": (
            sum(x["latency_seconds"] for x in examples)
            / num_examples
        ),
        "average_retrieval_calls": (
            sum(x["retrieval_calls"] for x in examples)
            / num_examples
        ),
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
        print(f"Answer coverage:   {metrics['answer_coverage']:.3f}")
        print(f"Exact Match:       {metrics['exact_match']:.3f}")
        print(f"F1:                {metrics['f1']:.3f}")
        print(f"Action accuracy:   {metrics['action_accuracy']:.3f}")
        print(f"Invalid outputs:   {metrics['invalid_output_rate']:.3f}")
        print(
            f"Decision ctx trunc: "
            f"{metrics['decision_context_truncated_rate']:.3f}"
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

    overall = evaluate_condition(results)

    action_accuracy = overall["action_accuracy"]
    invalid_output_rate = overall["invalid_output_rate"]

    # How often did C1 produce an answer when it should have abstained?
    # SEARCH always ends in an answer; INVALID executes as ABSTAIN.
    unanswerable = [
        x for x in results
        if x["condition"] == "unanswerable"
    ]

    false_answer_count = sum(
        x["final_action"] == "ANSWER"
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
        x["parsed_action"] == "SEARCH"
        for x in multi_hop
    ) / len(multi_hop)

    # How often did C1 unnecessarily search on Easy?
    easy = [
        x for x in results
        if x["condition"] == "easy"
    ]

    easy_search_rate = sum(
        x["parsed_action"] == "SEARCH"
        for x in easy
    ) / len(easy)

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
    print(f"Invalid-output rate:         {invalid_output_rate:.3f}")
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
        "input_path": str(input_path),
        "num_examples": len(results),
        "summary": summary,
        "overall": overall,
        "action_accuracy": action_accuracy,
        "invalid_output_rate": invalid_output_rate,
        "unanswerable_false_answer_rate": false_answer_rate,
        "multi_hop_search_rate": multi_hop_search_rate,
        "easy_search_rate": easy_search_rate,
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
