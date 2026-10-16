import sys
import unittest
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from actions import execution_action, parse_action  # noqa: E402
from prompt_budget import (  # noqa: E402
    MAX_INPUT_TOKENS,
    PromptTooLongError,
    count_tokens,
    fit_prompt,
)
import evaluate_c0  # noqa: E402
import evaluate_c1  # noqa: E402
import evaluate_c2  # noqa: E402


class WordTokenizer:
    """One token per whitespace word, plus one end-of-sequence token."""

    def __call__(self, text, add_special_tokens=True, verbose=True, **kwargs):
        ids = list(range(len(text.split())))

        if add_special_tokens:
            ids.append(-1)

        return {"input_ids": ids}


def build_context(documents):
    return "\n\n".join(
        f"[Document {i}: {doc['title']}]\n{doc['text']}"
        for i, doc in enumerate(documents, start=1)
    )


def build_prompt(question, context):
    return f"Choose ANSWER SEARCH or ABSTAIN.\nQuestion: {question}\nDocuments:\n{context}\nAction:"


def make_doc(title, num_words, score):
    return {"title": title, "text": " ".join(["word"] * num_words), "score": score}


class ParseActionTests(unittest.TestCase):

    def test_valid_actions_with_whitespace_and_punctuation(self):
        cases = {
            "ANSWER": "ANSWER",
            "  search \n": "SEARCH",
            "Abstain.": "ABSTAIN",
            "**ABSTAIN**": "ABSTAIN",
            "'answer'": "ANSWER",
            "SEARCH!": "SEARCH",
        }

        for raw, expected in cases.items():
            with self.subTest(raw=raw):
                self.assertEqual(parse_action(raw), expected)

    def test_unrecognized_outputs_are_invalid(self):
        for raw in [
            "",
            "   ",
            "maybe",
            "George Washington Bridge",
            "ANSWERS",
            "ANSWER because the documents say so",
            "I would SEARCH",
            "ANSWER SEARCH",
        ]:
            with self.subTest(raw=raw):
                self.assertEqual(parse_action(raw), "INVALID")

    def test_invalid_never_executes_as_answer(self):
        self.assertEqual(execution_action("INVALID"), "ABSTAIN")

        for action in ["ANSWER", "SEARCH", "ABSTAIN"]:
            self.assertEqual(execution_action(action), action)


class FitPromptTests(unittest.TestCase):

    def setUp(self):
        self.tokenizer = WordTokenizer()
        self.build = partial(build_prompt, "Who built the bridge?")

    def test_short_prompt_is_unchanged(self):
        docs = [make_doc("A", 10, 0.9), make_doc("B", 10, 0.8)]
        fitted = fit_prompt(self.tokenizer, self.build, build_context, docs, 100)

        self.assertFalse(fitted["context_truncated"])
        self.assertEqual(fitted["prompt"], self.build(build_context(docs)))
        self.assertEqual(fitted["documents_in_prompt"], 2)

    def test_drops_lowest_scoring_documents_and_keeps_cue(self):
        # Second-search documents can appear after first-search ones but
        # score higher; the lowest-scoring document must go first.
        docs = [
            make_doc("first_low", 30, 0.2),
            make_doc("second_high", 30, 0.9),
            make_doc("second_mid", 30, 0.5),
        ]
        fitted = fit_prompt(self.tokenizer, self.build, build_context, docs, 90)

        self.assertTrue(fitted["context_truncated"])
        self.assertEqual(fitted["titles_in_prompt"], ["second_high", "second_mid"])
        self.assertLessEqual(fitted["input_tokens"], 90)
        self.assertIn("Who built the bridge?", fitted["prompt"])
        self.assertIn("Choose ANSWER SEARCH or ABSTAIN.", fitted["prompt"])
        self.assertTrue(fitted["prompt"].endswith("Action:"))

    def test_trims_single_document_that_is_too_long(self):
        docs = [make_doc("Huge", 500, 0.9)]
        fitted = fit_prompt(self.tokenizer, self.build, build_context, docs, 60)

        self.assertTrue(fitted["context_truncated"])
        self.assertEqual(fitted["documents_in_prompt"], 1)
        self.assertEqual(fitted["input_tokens"], 60)
        self.assertTrue(fitted["prompt"].endswith("Action:"))

    def test_token_count_matches_full_prompt(self):
        docs = [make_doc(str(i), 40, 1 - i / 10) for i in range(5)]
        fitted = fit_prompt(self.tokenizer, self.build, build_context, docs, 120)

        self.assertEqual(
            fitted["input_tokens"],
            count_tokens(self.tokenizer, fitted["prompt"]),
        )

    def test_raises_when_fixed_text_cannot_fit(self):
        long_question = " ".join(["why"] * 200)
        build = partial(build_prompt, long_question)

        with self.assertRaises(PromptTooLongError) as ctx:
            fit_prompt(self.tokenizer, build, build_context, [make_doc("A", 5, 1.0)], 100)

        self.assertIn("Instructions and question alone", str(ctx.exception))


class EvaluationDenominatorTests(unittest.TestCase):

    def agent_record(self, condition, gold_action, parsed, final, answer, gold):
        return {
            "condition": condition,
            "gold_action": gold_action,
            "gold_answer": gold,
            "parsed_action": parsed,
            "first_action": parsed,
            "execution_action": "ABSTAIN" if parsed == "INVALID" else parsed,
            "final_action": final,
            "answer": answer,
            "decision_prompt": {"input_tokens": 100, "context_truncated": False},
            "latency_seconds": 1.0,
            "retrieval_calls": 1,
        }

    def test_abstentions_stay_in_em_f1_denominator(self):
        examples = [
            self.agent_record("easy", "ANSWER", "ANSWER", "ANSWER", "Paris", "Paris"),
            self.agent_record("easy", "ANSWER", "ABSTAIN", "ABSTAIN", "", "Paris"),
        ]

        for module in (evaluate_c1, evaluate_c2):
            with self.subTest(module=module.__name__):
                metrics = module.evaluate_condition(examples)

                self.assertEqual(metrics["num_examples"], 2)
                self.assertEqual(metrics["num_answered"], 1)
                self.assertEqual(metrics["answer_coverage"], 0.5)
                self.assertEqual(metrics["exact_match"], 0.5)
                self.assertEqual(metrics["f1"], 0.5)

    def test_empty_prediction_is_wrong_for_nonempty_gold(self):
        self.assertEqual(evaluate_c1.exact_match("", "the"), 0)
        self.assertEqual(evaluate_c1.token_f1("", "the"), 0)
        self.assertEqual(evaluate_c1.exact_match("   ", "yes"), 0)

    def test_invalid_counts_as_wrong_even_when_fallback_matches_gold(self):
        examples = [
            self.agent_record("unanswerable", "ABSTAIN", "INVALID", "ABSTAIN", "", "x"),
            self.agent_record("unanswerable", "ABSTAIN", "ABSTAIN", "ABSTAIN", "", "x"),
        ]

        for module in (evaluate_c1, evaluate_c2):
            with self.subTest(module=module.__name__):
                metrics = module.evaluate_condition(examples)

                self.assertEqual(metrics["action_accuracy"], 0.5)
                self.assertEqual(metrics["invalid_output_rate"], 0.5)
                self.assertEqual(metrics["answer_coverage"], 0.0)

    def test_c0_scores_every_example(self):
        examples = [
            {
                "answer": "Paris",
                "gold_answer": "Paris",
                "latency_seconds": 1.0,
                "output_tokens": 3,
                "input_tokens": 200,
                "context_truncated": False,
            },
            {
                "answer": "",
                "gold_answer": "Rome",
                "latency_seconds": 1.0,
                "output_tokens": 1,
                "input_tokens": 512,
                "context_truncated": True,
            },
        ]

        metrics = evaluate_c0.evaluate_group(examples)

        self.assertEqual(metrics["exact_match"], 0.5)
        self.assertEqual(metrics["answer_coverage"], 0.5)
        self.assertEqual(metrics["context_truncated_rate"], 0.5)


class RealTokenizerTests(unittest.TestCase):
    """Uses the locally cached flan-t5-base tokenizer; skipped if unavailable."""

    @classmethod
    def setUpClass(cls):
        try:
            from transformers import AutoTokenizer

            cls.tokenizer = AutoTokenizer.from_pretrained(
                "google/flan-t5-base",
                local_files_only=True,
            )
        except Exception as error:  # pragma: no cover - environment dependent
            raise unittest.SkipTest(f"flan-t5-base tokenizer unavailable: {error}")

    def long_documents(self):
        sentence = "The bridge spans the river between two boroughs of the city. "

        return [
            {"title": f"Doc {i}", "text": sentence * 40, "score": 1 - i / 10}
            for i in range(10)
        ]

    def test_all_prompt_builders_fit_512(self):
        import baseline_c0
        import c1_agent
        import c2_agent
        import run_c0_llm
        from prompt_budget import encode_prompt

        question = "Which borough is the western terminus of the bridge?"
        builders = {
            "c0_answer": (run_c0_llm.build_prompt, baseline_c0.build_context),
            "c1_decision": (c1_agent.build_decision_prompt, c1_agent.build_context),
            "c1_answer": (c1_agent.build_answer_prompt, c1_agent.build_context),
            "c2_decision": (c2_agent.build_sufficiency_prompt, c2_agent.build_context),
            "c2_answer": (c2_agent.build_answer_prompt, c2_agent.build_context),
        }

        for name, (builder, context_builder) in builders.items():
            with self.subTest(prompt=name):
                build = partial(builder, question)
                fitted = fit_prompt(
                    self.tokenizer,
                    build,
                    context_builder,
                    self.long_documents(),
                )

                self.assertTrue(fitted["context_truncated"])
                self.assertLessEqual(fitted["input_tokens"], MAX_INPUT_TOKENS)

                encoded = encode_prompt(self.tokenizer, fitted["prompt"])
                self.assertEqual(encoded["input_ids"].shape[1], fitted["input_tokens"])

                self.assertIn(question, fitted["prompt"])
                self.assertEqual(
                    fitted["prompt"].rstrip().split("\n")[-1],
                    build("").rstrip().split("\n")[-1],
                )

    def test_encode_prompt_refuses_overlong_input(self):
        from prompt_budget import encode_prompt

        with self.assertRaises(PromptTooLongError):
            encode_prompt(self.tokenizer, "word " * 700)


if __name__ == "__main__":
    unittest.main()
