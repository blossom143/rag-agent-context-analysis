
import argparse
import json
import time
from functools import partial
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from actions import execution_action, parse_action
from prompt_budget import MAX_INPUT_TOKENS, encode_prompt, fit_prompt
from retriever import DenseRetriever
from smoke import select_smoke_examples


BENCHMARK_PATH = Path("data/benchmark_60.json")
OUTPUT_PATH = Path("results/c1_results_v2.json")
SMOKE_OUTPUT_PATH = Path("results/smoke/c1_results_v2.json")

MODEL_NAME = "google/flan-t5-base"

TOP_K = 5
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
    inputs = encode_prompt(tokenizer, prompt, MAX_INPUT_TOKENS)

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


def fit_and_generate(
    build_prompt,
    documents,
    tokenizer,
    model,
    device,
    max_new_tokens,
):
    fitted = fit_prompt(
        tokenizer,
        build_prompt,
        build_context,
        documents,
        MAX_INPUT_TOKENS,
    )

    text = generate_text(
        fitted["prompt"],
        tokenizer,
        model,
        device,
        max_new_tokens,
    )

    return text, fitted


def prompt_stats(fitted):
    if fitted is None:
        return None

    return {
        "input_tokens": fitted["input_tokens"],
        "context_truncated": fitted["context_truncated"],
        "documents_retrieved": fitted["documents_retrieved"],
        "documents_in_prompt": fitted["documents_in_prompt"],
        "titles_in_prompt": fitted["titles_in_prompt"],
    }


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

    # =========================================================
    # STEP 2: Decide what to do
    # =========================================================

    decision_text, decision_fit = fit_and_generate(
        partial(build_decision_prompt, question),
        first_results,
        tokenizer,
        model,
        device,
        MAX_DECISION_TOKENS,
    )

    parsed_action = parse_action(decision_text)
    executed_action = execution_action(parsed_action)

    # =========================================================
    # STEP 3: Execute the decision
    # =========================================================

    retrieval_calls = 1

    second_results = []
    answer_fit = None

    if executed_action == "ANSWER":

        # Answer using the initial retrieved context.
        answer, answer_fit = fit_and_generate(
            partial(build_answer_prompt, question),
            first_results,
            tokenizer,
            model,
            device,
            MAX_ANSWER_TOKENS,
        )

        final_action = "ANSWER"

    elif executed_action == "SEARCH":

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

        # Answer after the second search.
        answer, answer_fit = fit_and_generate(
            partial(build_answer_prompt, question),
            combined_documents,
            tokenizer,
            model,
            device,
            MAX_ANSWER_TOKENS,
        )

        final_action = "ANSWER"

    else:
        # ABSTAIN (or the INVALID fallback) means no answer is generated.
        answer = ""
        final_action = "ABSTAIN"

    latency = time.perf_counter() - start_time

    return {
        "id": example["id"],
        "question": question,
        "condition": example["condition"],

        "gold_answer": example["answer"],
        "gold_action": example["gold_action"],

        # first_action is the parsed model decision and may be INVALID.
        "decision_text": decision_text,
        "parsed_action": parsed_action,
        "first_action": parsed_action,
        "execution_action": executed_action,
        "final_action": final_action,

        "decision_prompt": prompt_stats(decision_fit),
        "answer_prompt": prompt_stats(answer_fit),

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

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run one example per condition and write to results/smoke/.",
    )
    args = parser.parse_args()

    with open(BENCHMARK_PATH, "r") as f:
        benchmark = json.load(f)

    output_path = OUTPUT_PATH

    if args.smoke:
        benchmark = select_smoke_examples(benchmark)
        output_path = SMOKE_OUTPUT_PATH

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
            f"decision={result['parsed_action']:<8} "
            f"raw={result['decision_text'][:20]!r} "
            f"calls={result['retrieval_calls']} "
            f"tok={result['decision_prompt']['input_tokens']} "
            f"answer={result['answer'][:60]!r}"
        )

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
    print(f"Saved C1 results to: {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
