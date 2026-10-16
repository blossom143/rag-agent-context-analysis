
import argparse
import json
import time
from functools import partial
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from baseline_c0 import build_context
from prompt_budget import MAX_INPUT_TOKENS, encode_prompt, fit_prompt

INPUT_PATH = Path("results/c0_results_v2.json")
OUTPUT_PATH = Path("results/c0_answers_v2.json")
SMOKE_INPUT_PATH = Path("results/smoke/c0_results_v2.json")
SMOKE_OUTPUT_PATH = Path("results/smoke/c0_answers_v2.json")

MODEL_NAME = "google/flan-t5-base"
MAX_NEW_TOKENS = 64


def build_prompt(question, context):
    return f"""Answer the question using only the information in the documents below.
Give a short, direct answer. Do not use outside knowledge.

Question:
{question}

Documents:
{context}

Answer:"""


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


def generate_answer(question, documents, tokenizer, model, device):
    fitted = fit_prompt(
        tokenizer,
        partial(build_prompt, question),
        build_context,
        documents,
        MAX_INPUT_TOKENS,
    )

    inputs = encode_prompt(tokenizer, fitted["prompt"], MAX_INPUT_TOKENS)

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    input_token_count = inputs["input_ids"].shape[1]

    start_time = time.perf_counter()

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )

    latency = time.perf_counter() - start_time

    answer = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True,
    ).strip()

    output_token_count = outputs.shape[1]

    return {
        "answer": answer,
        "input_tokens": int(input_token_count),
        "output_tokens": int(output_token_count),
        "context_truncated": fitted["context_truncated"],
        "documents_in_prompt": fitted["documents_in_prompt"],
        "titles_in_prompt": fitted["titles_in_prompt"],
        "latency_seconds": latency,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Read and write the one-per-condition files in results/smoke/.",
    )
    args = parser.parse_args()

    input_path = SMOKE_INPUT_PATH if args.smoke else INPUT_PATH
    output_path = SMOKE_OUTPUT_PATH if args.smoke else OUTPUT_PATH

    if not input_path.exists():
        raise FileNotFoundError(
            f"{input_path} not found. Run baseline_c0.py"
            f"{' --smoke' if args.smoke else ''} first."
        )

    with open(input_path, "r") as f:
        examples = json.load(f)

    tokenizer, model, device = load_model()

    results = []

    print()
    print("=" * 70)
    print(f"Running C0 on {len(examples)} examples from {input_path}")
    print("=" * 70)

    for i, example in enumerate(examples, start=1):

        generation = generate_answer(
            question=example["question"],
            documents=example["retrieved_documents"],
            tokenizer=tokenizer,
            model=model,
            device=device,
        )

        result = {
            "id": example["id"],
            "question": example["question"],
            "condition": example["condition"],
            "gold_answer": example["gold_answer"],
            "gold_action": example["gold_action"],
            "predicted_action": "ANSWER",
            "retrieved_titles": example["retrieved_titles"],
            "answer": generation["answer"],
            "input_tokens": generation["input_tokens"],
            "output_tokens": generation["output_tokens"],
            "context_truncated": generation["context_truncated"],
            "documents_in_prompt": generation["documents_in_prompt"],
            "titles_in_prompt": generation["titles_in_prompt"],
            "latency_seconds": generation["latency_seconds"],
        }

        results.append(result)

        print(
            f"[{i:02d}/{len(examples)}] "
            f"{example['condition']:<12} "
            f"tok={generation['input_tokens']} "
            f"answer={generation['answer'][:80]!r} "
            f"time={generation['latency_seconds']:.2f}s"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print()
    print("=" * 70)
    print(f"Saved C0 answers to: {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
