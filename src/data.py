from datasets import load_dataset


def load_hotpotqa(split="train"):
    """Load HotpotQA distractor dataset."""
    return load_dataset(
        "hotpotqa/hotpot_qa",
        "distractor",
        split=split
    )


def get_context(example):
    """Convert HotpotQA context dictionary into a list of documents."""
    titles = example["context"]["title"]
    sentences = example["context"]["sentences"]

    return [
        {
            "title": title,
            "sentences": sents,
            "text": " ".join(sents)
        }
        for title, sents in zip(titles, sentences)
    ]


if __name__ == "__main__":
    dataset = load_hotpotqa("train")
    example = dataset[0]

    documents = get_context(example)

    print("Question:")
    print(example["question"])

    print("\nAnswer:")
    print(example["answer"])

    print("\nSupporting titles:")
    print(example["supporting_facts"]["title"])

    print("\nNumber of documents:", len(documents))

    for doc in documents:
        marker = " <-- SUPPORT" if doc["title"] in example["supporting_facts"]["title"] else ""

        print(f"\n{doc['title']}{marker}")
        print(doc["text"][:300])