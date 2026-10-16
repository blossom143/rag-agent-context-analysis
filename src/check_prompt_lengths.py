
import json
import sys
from functools import partial
from pathlib import Path

from transformers import AutoTokenizer

import baseline_c0
import c1_agent
import c2_agent
import run_c0_llm
from prompt_budget import MAX_INPUT_TOKENS, encode_prompt, fit_prompt
from retriever import DenseRetriever

BENCHMARK_PATH = Path("data/benchmark_60.json")
TOP_K = 5


def combine(first_results, second_results):
    combined = []
    seen_titles = set()

    for doc in first_results + second_results:
        if doc["title"] not in seen_titles:
            combined.append(doc)
            seen_titles.add(doc["title"])

    return combined


def main():
    """
    Fit every prompt the controllers can send (no generation) and
    confirm each full tokenization is within the model input limit.
    """
    with open(BENCHMARK_PATH, "r") as f:
        benchmark = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(c1_agent.MODEL_NAME)
    retriever = DenseRetriever()

    prompt_kinds = {
        "c0_answer": (run_c0_llm.build_prompt, baseline_c0.build_context, "first"),
        "c1_decision": (c1_agent.build_decision_prompt, c1_agent.build_context, "first"),
        "c1_answer_first": (c1_agent.build_answer_prompt, c1_agent.build_context, "first"),
        "c1_answer_search": (c1_agent.build_answer_prompt, c1_agent.build_context, "combined"),
        "c2_decision": (c2_agent.build_sufficiency_prompt, c2_agent.build_context, "first"),
        "c2_answer_first": (c2_agent.build_answer_prompt, c2_agent.build_context, "first"),
        "c2_answer_search": (c2_agent.build_answer_prompt, c2_agent.build_context, "combined"),
    }

    stats = {
        kind: {"max_tokens": 0, "truncated": 0, "untrimmed_over_limit": 0}
        for kind in prompt_kinds
    }
    failures = []

    for example in benchmark:
        question = example["question"]

        retriever.build_index(example["initial_documents"])
        first_results = retriever.search(question, top_k=TOP_K)

        retriever.build_index(example["documents"])
        second_results = retriever.search(question, top_k=TOP_K)

        documents = {
            "first": first_results,
            "combined": combine(first_results, second_results),
        }

        for kind, (builder, context_builder, source) in prompt_kinds.items():
            build = partial(builder, question)
            docs = documents[source]

            try:
                fitted = fit_prompt(
                    tokenizer,
                    build,
                    context_builder,
                    docs,
                    MAX_INPUT_TOKENS,
                )
                encoded = encode_prompt(tokenizer, fitted["prompt"], MAX_INPUT_TOKENS)
            except ValueError as error:
                failures.append(f"{example['id']} {kind}: {error}")
                continue

            length = encoded["input_ids"].shape[1]

            if length != fitted["input_tokens"]:
                failures.append(
                    f"{example['id']} {kind}: recorded {fitted['input_tokens']} "
                    f"tokens but encoded {length}"
                )

            untrimmed = len(
                tokenizer(build(context_builder(docs)), verbose=False)["input_ids"]
            )

            stats[kind]["max_tokens"] = max(stats[kind]["max_tokens"], length)
            stats[kind]["truncated"] += int(fitted["context_truncated"])
            stats[kind]["untrimmed_over_limit"] += int(untrimmed > MAX_INPUT_TOKENS)

    print()
    print("=" * 70)
    print(f"PROMPT LENGTH CHECK (limit {MAX_INPUT_TOKENS} tokens, {len(benchmark)} examples)")
    print("=" * 70)

    for kind, values in stats.items():
        print(
            f"{kind:<18} max={values['max_tokens']:>3} "
            f"context_trimmed={values['truncated']:>2}/{len(benchmark)} "
            f"untrimmed_over_limit={values['untrimmed_over_limit']:>2}"
        )

    if failures:
        print()
        print(f"FAIL: {len(failures)} prompts")

        for failure in failures:
            print(f"  - {failure}")

        sys.exit(1)

    print()
    print("PASS: every fitted prompt is within the limit with no tokenizer truncation.")


if __name__ == "__main__":
    main()
