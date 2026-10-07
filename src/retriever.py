import json
import numpy as np
import torch
from sentence_transformers import SentenceTransformer


MODEL_NAME = "BAAI/bge-small-en-v1.5"


class DenseRetriever:
    def __init__(self, model_name=MODEL_NAME, model=None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if model is not None:
            # Reuse an already-loaded embedding model
            self.model = model
            print("Using already-loaded embedding model.")
        else:
            # Load the model normally
            print(f"Loading embedding model: {model_name}")
            print(f"Device: {self.device}")

            self.model = SentenceTransformer(
                model_name,
                device=self.device,
            )

        self.documents = []
        self.embeddings = None

    def build_index(self, documents):
        """
        Encode all documents and store their embeddings.

        documents: list of dictionaries containing:
            - title
            - text
        """
        self.documents = documents

        texts = [
            f"{doc['title']}: {doc['text']}"
            for doc in documents
        ]

        print(f"Encoding {len(texts)} documents...")

        self.embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

        print("Index built.")

    def search(self, query, top_k=5):
        """
        Return the top-k most similar documents.
        """
        if self.embeddings is None:
            raise RuntimeError("Build the index before searching.")

        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )[0]

        scores = self.embeddings @ query_embedding

        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []

        for idx in top_indices:
            results.append({
                "title": self.documents[idx]["title"],
                "text": self.documents[idx]["text"],
                "score": float(scores[idx]),
            })

        return results


def load_benchmark(path="data/benchmark_60.json"):
    with open(path, "r") as f:
        return json.load(f)


def main():
    benchmark = load_benchmark()

    # For the first test, use documents from one example.
    example = benchmark[0]

    retriever = DenseRetriever()

    retriever.build_index(example["initial_documents"])

    results = retriever.search(
        example["question"],
        top_k=5,
    )

    print("\n" + "=" * 60)
    print("QUESTION")
    print("=" * 60)
    print(example["question"])

    print("\n" + "=" * 60)
    print("TOP RETRIEVED DOCUMENTS")
    print("=" * 60)

    for rank, result in enumerate(results, start=1):
        print(f"\nRank {rank}")
        print(f"Title: {result['title']}")
        print(f"Score: {result['score']:.4f}")
        print(f"Text: {result['text'][:300]}")


if __name__ == "__main__":
    main()