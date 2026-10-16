import argparse
import json
from pathlib import Path

from retriever import DenseRetriever
from smoke import select_smoke_examples


BENCHMARK_PATH = Path("data/benchmark_60.json")
OUTPUT_PATH = Path("results/c0_results_v2.json")
SMOKE_OUTPUT_PATH = Path("results/smoke/c0_results_v2.json")

TOP_K = 5


def build_context(documents):
    """Convert retrieved documents into an LLM-readable context."""

    context_parts = []

    for i, doc in enumerate(documents, start=1):
        context_parts.append(
            f"[Document {i}: {doc['title']}]\n"
            f"{doc['text']}"
        )

    return "\n\n".join(context_parts)


def evaluate_example(example, retriever):
    """
    Retrieve the top-k documents available to the agent.

    This function currently implements the retrieval portion
    of C0. The answer-generation step will be added after
    retrieval is verified.
    """

    documents = example["initial_documents"]

    retriever.build_index(documents)

    retrieved_docs = retriever.search(
        example["question"],
        top_k=TOP_K,
    )

    context = build_context(retrieved_docs)

    return {
        "id": example["id"],
        "question": example["question"],
        "condition": example["condition"],
        "gold_answer": example["answer"],
        "gold_action": example["gold_action"],
        "retrieved_titles": [
            doc["title"]
            for doc in retrieved_docs
        ],
        # Ranked documents with scores, so answer generation can fit
        # them to the model's input limit. "context" is the untrimmed text.
        "retrieved_documents": retrieved_docs,
        "context": context,
    }


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run one example per condition and write to results/smoke/.",
    )
    args = parser.parse_args()

    # ---------------------------------------------------------
    # Load benchmark
    # ---------------------------------------------------------

    with open(BENCHMARK_PATH, "r") as f:
        benchmark = json.load(f)

    output_path = OUTPUT_PATH

    if args.smoke:
        benchmark = select_smoke_examples(benchmark)
        output_path = SMOKE_OUTPUT_PATH

    print(f"Evaluating {len(benchmark)} examples...")

    # ---------------------------------------------------------
    # Create retriever once
    # ---------------------------------------------------------

    retriever = DenseRetriever()

    results = []

    # ---------------------------------------------------------
    # Retrieve context for every example
    # ---------------------------------------------------------

    for i, example in enumerate(benchmark, start=1):

        result = evaluate_example(
            example,
            retriever,
        )

        results.append(result)

        print(
            f"[{i:02d}/{len(benchmark)}] "
            f"{example['condition']:<12} "
            f"retrieved={len(result['retrieved_titles'])}"
        )

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(output_path, "w") as f:
        json.dump(
            results,
            f,
            indent=2,
        )

    print()
    print("=" * 70)
    print(f"Saved C0 retrieval contexts to: {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()