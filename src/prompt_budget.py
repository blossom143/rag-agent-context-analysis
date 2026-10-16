
# google/flan-t5-base was trained with 512-token inputs. The count includes
# the end-of-sequence token the tokenizer appends.
MAX_INPUT_TOKENS = 512


class PromptTooLongError(ValueError):
    pass


def count_tokens(tokenizer, text):
    return len(tokenizer(text, add_special_tokens=True, verbose=False)["input_ids"])


def _trim_text(tokenizer, build_prompt, build_context, documents, last_index, max_tokens):
    """Keep the longest word prefix of documents[last_index] that fits."""
    words = documents[last_index]["text"].split()
    low, high = 0, len(words)

    def candidate(num_words):
        trimmed = dict(documents[last_index], text=" ".join(words[:num_words]))
        return documents[:last_index] + [trimmed]

    while low < high:
        mid = (low + high + 1) // 2
        prompt = build_prompt(build_context(candidate(mid)))

        if count_tokens(tokenizer, prompt) <= max_tokens:
            low = mid
        else:
            high = mid - 1

    if low == 0:
        return documents[:last_index]

    return candidate(low)


def fit_prompt(
    tokenizer,
    build_prompt,
    build_context,
    documents,
    max_tokens=MAX_INPUT_TOKENS,
):
    """
    Build a prompt whose full tokenization fits in max_tokens.

    build_prompt(context) must return the complete prompt for a context
    string, so the question, instructions, and final cue are always kept.
    Only retrieved context is shortened: the lowest-scoring documents are
    dropped first, and the single remaining document is trimmed by words
    only if it still does not fit. Document order is otherwise preserved.
    """
    fixed_tokens = count_tokens(tokenizer, build_prompt(""))

    if fixed_tokens > max_tokens:
        raise PromptTooLongError(
            f"Instructions and question alone use {fixed_tokens} tokens, "
            f"over the {max_tokens}-token input limit."
        )

    kept = list(documents)
    prompt = build_prompt(build_context(kept))
    truncated = False

    # Rank positions, lowest score first; stable so ties drop the later document.
    drop_order = sorted(
        range(len(kept)),
        key=lambda i: kept[i].get("score", -i),
    )

    while count_tokens(tokenizer, prompt) > max_tokens and len(kept) > 1:
        truncated = True
        drop_position = drop_order.pop(0)
        drop_order = [i - (i > drop_position) for i in drop_order]
        kept.pop(drop_position)
        prompt = build_prompt(build_context(kept))

    if count_tokens(tokenizer, prompt) > max_tokens:
        truncated = True
        kept = _trim_text(
            tokenizer,
            build_prompt,
            build_context,
            kept,
            len(kept) - 1,
            max_tokens,
        )
        prompt = build_prompt(build_context(kept))

    input_tokens = count_tokens(tokenizer, prompt)

    if input_tokens > max_tokens:
        raise PromptTooLongError(
            f"Fitted prompt still uses {input_tokens} tokens "
            f"(limit {max_tokens})."
        )

    return {
        "prompt": prompt,
        "input_tokens": input_tokens,
        "context_truncated": truncated,
        "documents_retrieved": len(documents),
        "documents_in_prompt": len(kept),
        "titles_in_prompt": [doc["title"] for doc in kept],
    }


def encode_prompt(tokenizer, prompt, max_tokens=MAX_INPUT_TOKENS):
    """Tokenize without truncation and refuse anything over the limit."""
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=False,
        verbose=False,
    )

    length = inputs["input_ids"].shape[1]

    if length > max_tokens:
        raise PromptTooLongError(
            f"Prompt has {length} tokens, over the {max_tokens}-token limit. "
            "Build prompts with fit_prompt()."
        )

    return inputs
