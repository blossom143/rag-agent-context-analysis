import json


with open("data/benchmark_60.json", "r") as f:
    benchmark = json.load(f)


for condition in ["easy", "multi_hop", "unanswerable"]:
    examples = [x for x in benchmark if x["condition"] == condition]

    print("\n" + "=" * 70)
    print(f"{condition.upper()} ({len(examples)} examples)")
    print("=" * 70)

    for i, example in enumerate(examples[:3]):
        print(f"\nExample {i + 1}")
        print("-" * 50)

        print("ID:", example["id"])
        print("Question:", example["question"])
        print("Answer:", example["answer"])
        print("Gold action:", example["gold_action"])

        print("\nInitial documents:")
        for doc in example["initial_documents"]:
            print("  -", doc["title"])

        if condition == "multi_hop":
            print("\nSearch reveals:")
            print("  -", example["search_reveal"]["title"])