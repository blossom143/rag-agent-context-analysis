
import json
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

INPUT_PATH = Path("results/c0_results.json")
OUTPUT_PATH = Path("results/c0_answers.json")

MODEL_NAME = "google/flan-t5-base"
MAX_INPUT_LENGTH = 1024
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


def generate_answer(question, context, tokenizer, model, device):
    prompt = build_prompt(question, context)

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
        "latency_seconds": latency,
    }


def main():
    with open(INPUT_PATH, "r") as f:
        examples = json.load(f)

    tokenizer, model, device = load_model()

    results = []

    print()
    print("=" * 70)
    print(f"Running C0 on {len(examples)} examples")
    print("=" * 70)

    for i, example in enumerate(examples, start=1):

        generation = generate_answer(
            question=example["question"],
            context=example["context"],
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
            "latency_seconds": generation["latency_seconds"],
        }

        results.append(result)

        print(
            f"[{i:02d}/{len(examples)}] "
            f"{example['condition']:<12} "
            f"answer={generation['answer'][:80]!r} "
            f"time={generation['latency_seconds']:.2f}s"
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print()
    print("=" * 70)
    print(f"Saved C0 answers to: {OUTPUT_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()
