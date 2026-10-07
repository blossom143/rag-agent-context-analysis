import json
from pathlib import Path

from retriever import DenseRetriever


BENCHMARK_PATH = Path("data/benchmark_60.json")
RESULTS_DIR = Path("results")
OUTPUT_PATH = RESULTS_DIR / "c0_results.json"

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
        "context": context,
    }


def main():

    # ---------------------------------------------------------
    # Load benchmark
    # ---------------------------------------------------------

    with open(BENCHMARK_PATH, "r") as f:
        benchmark = json.load(f)

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

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(OUTPUT_PATH, "w") as f:
        json.dump(
            results,
            f,
            indent=2,
        )

    print()
    print("=" * 70)
    print(f"Saved C0 retrieval contexts to: {OUTPUT_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()