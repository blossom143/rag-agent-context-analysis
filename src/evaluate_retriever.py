import json
from pathlib import Path

from retriever import DenseRetriever


BENCHMARK_PATH = Path("data/benchmark_60.json")
RESULTS_DIR = Path("results")
OUTPUT_PATH = RESULTS_DIR / "retrieval_results.json"

TOP_K = 5


def evaluate_example(example, retriever):
    """
    Build a retrieval index over the documents initially available
    to the agent, then retrieve the top-k documents for the question.
    """

    documents = example["initial_documents"]

    retriever.build_index(documents)

    retrieved_docs = retriever.search(
        example["question"],
        top_k=TOP_K,
    )

    retrieved_titles = [
        doc["title"]
        for doc in retrieved_docs
    ]

    condition = example["condition"]

    # Unanswerable examples intentionally have no available
    # supporting evidence, so retrieval recall is not defined.
    if condition == "unanswerable":
        recall = None

    elif condition == "multi_hop":
        supporting_titles = set(example["supporting_titles"])

        # Only supporting documents that are actually available
        # in the initial context can be retrieved at this stage.
        available_supporting_titles = {
            title
            for title in supporting_titles
            if any(doc["title"] == title for doc in documents)
        }

        if len(available_supporting_titles) == 0:
            recall = 0.0
        else:
            retrieved_support = (
                set(retrieved_titles)
                & available_supporting_titles
            )

            recall = (
                len(retrieved_support)
                / len(available_supporting_titles)
            )

    else:
        # Easy condition: all supporting documents are initially
        # available to the retriever.
        supporting_titles = set(example["supporting_titles"])

        retrieved_support = (
            set(retrieved_titles)
            & supporting_titles
        )

        recall = (
            len(retrieved_support)
            / len(supporting_titles)
        )

    return retrieved_titles, recall


def main():

    # ---------------------------------------------------------
    # Load benchmark
    # ---------------------------------------------------------

    with open(BENCHMARK_PATH, "r") as f:
        benchmark = json.load(f)

    print(f"Evaluating {len(benchmark)} examples...")

    # ---------------------------------------------------------
    # Create ONE retriever.
    #
    # DenseRetriever loads BAAI/bge-small-en-v1.5 internally.
    # ---------------------------------------------------------

    retriever = DenseRetriever()

    results = []

    # ---------------------------------------------------------
    # Evaluate every example
    # ---------------------------------------------------------

    for i, example in enumerate(benchmark, start=1):

        retrieved_titles, recall = evaluate_example(
            example,
            retriever,
        )

        result = {
            "id": example["id"],
            "question": example["question"],
            "condition": example["condition"],
            "gold_action": example["gold_action"],
            "supporting_titles": example.get(
                "supporting_titles",
                []
            ),
            "retrieved_titles": retrieved_titles,
            "recall_at_5": recall,
        }

        results.append(result)

        if recall is None:
            recall_string = "None"
        else:
            recall_string = f"{recall:.1f}"

        print(
            f"[{i:02d}/{len(benchmark)}] "
            f"{example['condition']:<12} "
            f"recall={recall_string}"
        )

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    print()
    print("=" * 70)
    print("RETRIEVAL SUMMARY")
    print("=" * 70)

    summary = {}

    for condition in [
        "easy",
        "multi_hop",
        "unanswerable",
    ]:

        condition_results = [
            r
            for r in results
            if r["condition"] == condition
        ]

        valid_recalls = [
            r["recall_at_5"]
            for r in condition_results
            if r["recall_at_5"] is not None
        ]

        print()
        print(condition)
        print(f"Examples: {len(condition_results)}")

        if valid_recalls:

            mean_recall = (
                sum(valid_recalls)
                / len(valid_recalls)
            )

            print(
                f"Mean Recall@5: "
                f"{mean_recall:.3f}"
            )

            summary[condition] = {
                "num_examples": len(condition_results),
                "mean_recall_at_5": mean_recall,
            }

        else:

            print("Recall: N/A")

            summary[condition] = {
                "num_examples": len(condition_results),
                "mean_recall_at_5": None,
            }

    # ---------------------------------------------------------
    # Save results
    # ---------------------------------------------------------

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = {
        "model": "BAAI/bge-small-en-v1.5",
        "top_k": TOP_K,
        "num_examples": len(benchmark),
        "summary": summary,
        "examples": results,
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
        f"Saved results to: "
        f"{OUTPUT_PATH}"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()