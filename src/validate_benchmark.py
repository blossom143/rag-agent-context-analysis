
import json
import sys
from pathlib import Path
from collections import Counter

BENCHMARK_PATH = Path("data/benchmark_60.json")

EXPECTED_COUNTS = {
    "easy": 20,
    "multi_hop": 20,
    "unanswerable": 20,
}

REQUIRED_FIELDS = (
    "id",
    "question",
    "answer",
    "condition",
    "gold_action",
    "documents",
    "initial_documents",
    "supporting_titles",
)

REQUIRED_DOCUMENT_FIELDS = ("title", "text")


def validate_structure(benchmark):
    """Check fields, ids, and the condition distribution."""
    errors = []

    for index, example in enumerate(benchmark):
        label = example.get("id", f"index {index}")

        missing = [
            field for field in REQUIRED_FIELDS
            if field not in example
        ]

        if missing:
            errors.append(f"{label}: missing fields {missing}")
            continue

        for field in ("id", "question", "answer"):
            if not isinstance(example[field], str) or not example[field].strip():
                errors.append(f"{label}: field {field!r} is empty")

        for list_field in ("documents", "initial_documents"):
            for doc_index, doc in enumerate(example[list_field]):
                missing_doc = [
                    field for field in REQUIRED_DOCUMENT_FIELDS
                    if field not in doc
                ]

                if missing_doc:
                    errors.append(
                        f"{label}: {list_field}[{doc_index}] "
                        f"missing fields {missing_doc}"
                    )

    ids = [example.get("id") for example in benchmark]
    duplicates = sorted(
        str(item) for item, count in Counter(ids).items()
        if count > 1
    )

    if duplicates:
        errors.append(f"duplicate ids: {duplicates}")

    counts = Counter(example.get("condition") for example in benchmark)

    if dict(counts) != EXPECTED_COUNTS:
        errors.append(
            f"condition counts {dict(counts)} != expected {EXPECTED_COUNTS}"
        )

    return errors


def validate_example(example):
    errors = []

    condition = example["condition"]
    documents = example["documents"]
    initial_documents = example["initial_documents"]
    supporting_titles = set(example["supporting_titles"])
    gold_action = example["gold_action"]

    document_titles = {doc["title"] for doc in documents}
    initial_titles = {doc["title"] for doc in initial_documents}

    # For Easy and Multi-hop examples, supporting documents
    # must exist in the candidate pool.
    #
    # For Unanswerable examples, the supporting documents
    # are intentionally removed from the candidate pool.
    if condition in {"easy", "multi_hop"}:
        if not supporting_titles.issubset(document_titles):
            missing = supporting_titles - document_titles
            errors.append(
                f"supporting documents missing from candidate pool: {missing}"
            )

    # Initial documents must always be a subset of the candidate pool.
    if not initial_titles.issubset(document_titles):
        missing = initial_titles - document_titles
        errors.append(
            f"initial documents missing from candidate pool: {missing}"
        )

    if condition == "easy":

        missing_support = supporting_titles - initial_titles

        if missing_support:
            errors.append(
                f"easy example missing initial support: {missing_support}"
            )

        if gold_action != "ANSWER":
            errors.append(
                f"easy example has gold action {gold_action}, expected ANSWER"
            )

    elif condition == "multi_hop":

        available_support = supporting_titles & initial_titles
        missing_support = supporting_titles - initial_titles

        # At least one supporting document should be available initially.
        if len(available_support) == 0:
            errors.append(
                "multi-hop example has no supporting document initially available"
            )

        # At least one supporting document should be held out initially.
        if len(missing_support) == 0:
            errors.append(
                "multi-hop example has all supporting documents initially available"
            )

        if gold_action != "SEARCH":
            errors.append(
                f"multi-hop example has gold action {gold_action}, expected SEARCH"
            )

    elif condition == "unanswerable":

        available_support = supporting_titles & initial_titles

        # Both supporting documents should be absent from the initial context.
        if available_support:
            errors.append(
                f"unanswerable example still contains support: {available_support}"
            )

        # The supporting documents are intentionally absent from the
        # candidate pool as well.
        if supporting_titles & document_titles:
            errors.append(
                "unanswerable example contains supporting documents in candidate pool"
            )

        if gold_action != "ABSTAIN":
            errors.append(
                f"unanswerable example has gold action {gold_action}, expected ABSTAIN"
            )

    else:
        errors.append(f"unknown condition: {condition}")

    return errors


def main():

    with open(BENCHMARK_PATH, "r") as f:
        benchmark = json.load(f)

    print("=" * 70)
    print("BENCHMARK VALIDATION")
    print("=" * 70)

    print(f"Total examples: {len(benchmark)}")

    conditions = ["easy", "multi_hop", "unanswerable"]

    for condition in conditions:
        count = sum(
            1
            for example in benchmark
            if example["condition"] == condition
        )

        print(f"{condition:<15}: {count}")

    # Check unique IDs
    ids = [example["id"] for example in benchmark]

    print(f"Unique IDs:       {len(set(ids))}")

    if len(ids) != len(set(ids)):
        print("WARNING: Duplicate IDs found.")

    print()
    print("=" * 70)
    print("STRUCTURE VALIDATION")
    print("=" * 70)

    structure_errors = validate_structure(benchmark)

    if structure_errors:
        print(f"FAIL: {len(structure_errors)} structural errors.")

        for error in structure_errors:
            print(f"  - {error}")

        print()
        print("FINAL VERDICT: BENCHMARK NEEDS REVISION")
        sys.exit(1)

    print("PASS: required fields present, ids unique, 20/20/20 conditions.")

    # Validate individual examples
    all_errors = []

    for example in benchmark:

        errors = validate_example(example)

        if errors:
            all_errors.append(
                {
                    "id": example["id"],
                    "condition": example["condition"],
                    "errors": errors,
                }
            )

    print()
    print("=" * 70)
    print("CONDITION VALIDATION")
    print("=" * 70)

    if not all_errors:

        print("PASS: All 60 examples satisfy their intended condition.")

    else:

        print(
            f"FAIL: {len(all_errors)} examples contain validation errors."
        )

        for item in all_errors:

            print()
            print(f"ID: {item['id']}")
            print(f"Condition: {item['condition']}")

            for error in item["errors"]:
                print(f"  - {error}")

    # Supporting document availability
    print()
    print("=" * 70)
    print("SUPPORTING DOCUMENT AVAILABILITY")
    print("=" * 70)

    for condition in conditions:

        examples = [
            example
            for example in benchmark
            if example["condition"] == condition
        ]

        availability_counts = []

        for example in examples:

            supporting_titles = set(
                example["supporting_titles"]
            )

            initial_titles = {
                doc["title"]
                for doc in example["initial_documents"]
            }

            available = len(
                supporting_titles & initial_titles
            )

            total = len(supporting_titles)

            availability_counts.append(
                f"{available}/{total}"
            )

        distribution = Counter(availability_counts)

        print()
        print(condition)

        for availability, count in sorted(
            distribution.items()
        ):
            print(
                f"  {availability}: {count} examples"
            )

    # Document counts
    print()
    print("=" * 70)
    print("DOCUMENT COUNTS")
    print("=" * 70)

    for condition in conditions:

        examples = [
            example
            for example in benchmark
            if example["condition"] == condition
        ]

        candidate_counts = [
            len(example["documents"])
            for example in examples
        ]

        initial_counts = [
            len(example["initial_documents"])
            for example in examples
        ]

        print()
        print(condition)

        print(
            f"  Candidate documents: "
            f"min={min(candidate_counts)}, "
            f"max={max(candidate_counts)}, "
            f"avg={sum(candidate_counts) / len(candidate_counts):.1f}"
        )

        print(
            f"  Initial documents:   "
            f"min={min(initial_counts)}, "
            f"max={max(initial_counts)}, "
            f"avg={sum(initial_counts) / len(initial_counts):.1f}"
        )

    print()
    print("=" * 70)

    if not all_errors:

        print("FINAL VERDICT: BENCHMARK PASSED")

    else:

        print("FINAL VERDICT: BENCHMARK NEEDS REVISION")

    print("=" * 70)

    if all_errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
