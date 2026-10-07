
import json
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from retriever import DenseRetriever


BENCHMARK_PATH = Path("data/benchmark_60.json")
OUTPUT_PATH = Path("results/c1_results.json")

MODEL_NAME = "google/flan-t5-base"

TOP_K = 5
MAX_INPUT_LENGTH = 1024
MAX_DECISION_TOKENS = 16
MAX_ANSWER_TOKENS = 64


def build_context(documents):
    parts = []

    for i, doc in enumerate(documents, start=1):
        parts.append(
            f"[Document {i}: {doc['title']}]\n"
            f"{doc['text']}"
        )

    return "\n\n".join(parts)


def build_decision_prompt(question, context):
    return f"""You are a retrieval agent.

You must choose exactly one action:

ANSWER
SEARCH
ABSTAIN

ANSWER = the documents contain enough evidence to answer the question.

SEARCH = the documents do not contain enough evidence, but another search may find the missing information.

ABSTAIN = the question cannot be answered from the available evidence.

Question:
{question}

Documents:
{context}

Action:"""


def build_answer_prompt(question, context):
    return f"""Answer the question using only the information in the documents.

Give a short, direct answer.
Do not use outside knowledge.

Question:
{question}

Documents:
{context}

Answer:"""


def generate_text(
    prompt,
    tokenizer,
    model,
    device,
    max_new_tokens,
):
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_LENGTH,
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    return tokenizer.decode(
        outputs[0],
        skip_special_tokens=True,
    ).strip()


def parse_action(text):
    text = text.upper().strip()

    # Prefer exact first-token decisions.
    first_word = text.split()[0] if text.split() else ""

    if first_word in {"ANSWER", "SEARCH", "ABSTAIN"}:
        return first_word

    # Fallback if the model produces extra text.
    if "ABSTAIN" in text:
        return "ABSTAIN"

    if "SEARCH" in text:
        return "SEARCH"

    if "ANSWER" in text:
        return "ANSWER"

    # Conservative default.
    return "ANSWER"


def load_model():
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print(f"Loading model: {MODEL_NAME}")
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    model.to(device)
    model.eval()

    return tokenizer, model, device


def evaluate_example(
    example,
    retriever,
    tokenizer,
    model,
    device,
):
    start_time = time.perf_counter()

    question = example["question"]

    # =========================================================
    # STEP 1: Initial retrieval
    # =========================================================

    initial_documents = example["initial_documents"]

    retriever.build_index(initial_documents)

    first_results = retriever.search(
        question,
        top_k=TOP_K,
    )

    first_context = build_context(first_results)

    # =========================================================
    # STEP 2: Decide what to do
    # =========================================================

    decision_prompt = build_decision_prompt(
        question,
        first_context,
    )

    decision_text = generate_text(
        decision_prompt,
        tokenizer,
        model,
        device,
        MAX_DECISION_TOKENS,
    )

    first_action = parse_action(decision_text)

    # =========================================================
    # STEP 3: Execute the decision
    # =========================================================

    retrieval_calls = 1

    second_results = []

    if first_action == "ANSWER":

        # Answer using the initial retrieved context.
        answer_prompt = build_answer_prompt(
            question,
            first_context,
        )

        answer = generate_text(
            answer_prompt,
            tokenizer,
            model,
            device,
            MAX_ANSWER_TOKENS,
        )

        final_action = "ANSWER"

    elif first_action == "SEARCH":

        # Search the full candidate pool.
        retrieval_calls += 1

        full_documents = example["documents"]

        retriever.build_index(full_documents)

        second_results = retriever.search(
            question,
            top_k=TOP_K,
        )

        # Combine first and second retrieval results.
        combined_documents = []
        seen_titles = set()

        for doc in first_results + second_results:
            if doc["title"] not in seen_titles:
                combined_documents.append(doc)
                seen_titles.add(doc["title"])

        final_context = build_context(combined_documents)

        # Answer after the second search.
        answer_prompt = build_answer_prompt(
            question,
            final_context,
        )

        answer = generate_text(
            answer_prompt,
            tokenizer,
            model,
            device,
            MAX_ANSWER_TOKENS,
        )

        final_action = "ANSWER"

    else:
        # ABSTAIN means no answer is generated.
        answer = ""
        final_action = "ABSTAIN"

    latency = time.perf_counter() - start_time

    return {
        "id": example["id"],
        "question": question,
        "condition": example["condition"],

        "gold_answer": example["answer"],
        "gold_action": example["gold_action"],

        "first_action": first_action,
        "final_action": final_action,

        "decision_text": decision_text,

        "answer": answer,

        "retrieval_calls": retrieval_calls,

        "first_retrieved_titles": [
            doc["title"]
            for doc in first_results
        ],

        "second_retrieved_titles": [
            doc["title"]
            for doc in second_results
        ],

        "latency_seconds": latency,
    }


def main():

    with open(BENCHMARK_PATH, "r") as f:
        benchmark = json.load(f)

    tokenizer, model, device = load_model()

    retriever = DenseRetriever()

    results = []

    print()
    print("=" * 70)
    print(f"Running C1 on {len(benchmark)} examples")
    print("=" * 70)

    for i, example in enumerate(benchmark, start=1):

        result = evaluate_example(
            example,
            retriever,
            tokenizer,
            model,
            device,
        )

        results.append(result)

        print(
            f"[{i:02d}/{len(benchmark)}] "
            f"{example['condition']:<12} "
            f"decision={result['first_action']:<8} "
            f"calls={result['retrieval_calls']} "
            f"answer={result['answer'][:60]!r}"
        )

    OUTPUT_PATH.parent.mkdir(
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
    print(f"Saved C1 results to: {OUTPUT_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()
