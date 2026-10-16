# Answer, Search, or Abstain: Can RAG Agents Tell When Their Context Is Enough?

## Overview

Retrieval-Augmented Generation (RAG) systems combine information retrieval with language generation to answer questions using external documents. However, retrieving relevant documents does not guarantee that the retrieved context contains enough evidence to answer a question correctly.

This project investigates whether lightweight LLM-based controllers can determine when to answer using the available context, retrieve additional evidence, or abstain when the answer cannot be supported.

Three approaches are evaluated under a controlled benchmark containing answerable, multi-hop, and unanswerable questions.

## Research Question

**Can an LLM-based RAG controller reliably distinguish between sufficient context, incomplete context, and unavailable evidence?**

The project evaluates whether explicit action instructions and a context-sufficiency checklist improve decision-making compared with a baseline that always generates an answer.

## Methodology

The experimental pipeline consists of document retrieval, controller decision-making, answer generation, and evaluation.

1. **Retrieve:** Retrieve candidate documents relevant to the input question.
2. **Assess:** Determine whether the current context contains sufficient evidence.
3. **Act:** Select an action based on the controller strategy.
4. **Execute:** Generate an answer, perform another retrieval, or abstain.
5. **Evaluate:** Measure answer quality, action-selection accuracy, coverage, and reliability.

The available actions are:

- **ANSWER:** Generate an answer from the current evidence.
- **SEARCH:** Retrieve additional documents because the current context may be incomplete.
- **ABSTAIN:** Decline to answer when the necessary evidence is unavailable.

## Experimental Setup

| Component | Configuration |
|---|---|
| Benchmark | 60 controlled question instances |
| Easy questions | 20 |
| Multi-hop questions | 20 |
| Unanswerable questions | 20 |
| Dense retrieval model | `BAAI/bge-small-en-v1.5` |
| Generator and controller | `google/flan-t5-base` |
| Retrieval depth | Top 5 documents |
| Execution environment | Google Colab with T4 GPU |

The benchmark is derived from the HotpotQA validation distractor split and organized into three experimental conditions.

- **Easy:** Supporting evidence is available in the initial context.
- **Multi-hop:** The initial context contains incomplete evidence, while the missing supporting document remains available in the candidate pool.
- **Unanswerable:** The required supporting documents are removed from the candidate pool, making recovery through additional retrieval impossible under the benchmark setup.

These conditions are controlled experimental constructions rather than native HotpotQA labels.

## Controller Strategies

### C0 — Always-Answer Baseline

The baseline retrieves documents once and generates an answer without an explicit sufficiency decision. It provides a reference point for assessing whether decision-oriented controllers improve reliability.

### C1 — Explicit Action Prompt

C1 instructs the language model to select among `ANSWER`, `SEARCH`, and `ABSTAIN`. A `SEARCH` decision triggers another retrieval operation.

### C2 — Sufficiency Checklist

C2 uses a structured checklist to assess whether the current evidence is sufficient before choosing an action. It is designed to encourage more deliberate evidence assessment than a direct action prompt.

## Evaluation Metrics

The controllers are evaluated using answer-quality and decision-quality metrics.

| Metric | Description |
|---|---|
| Exact Match (EM) | Fraction of predictions that exactly match the reference answer after normalization |
| Token-level F1 | Overlap between predicted and reference answer tokens |
| Answer coverage | Fraction of questions for which the system produces an answer |
| First-action accuracy | Fraction of questions for which the initial action matches the intended action for that condition |
| Invalid decision rate | Fraction of controller outputs that cannot be parsed into a valid action |
| False-answer rate | Fraction of unanswerable questions for which the system produces an answer instead of abstaining |
| Average latency | Average execution time per question |
| Average retrieval calls | Average number of retrieval operations per question |

Answer quality and action-selection accuracy are evaluated separately because a correct answer does not necessarily imply that the controller selected the appropriate action.

## Results

The following results are from the corrected v2 experiment.

| Metric | C0 | C1 | C2 |
|---|---:|---:|---:|
| Overall EM | 31.7% | 30.0% | 26.7% |
| Overall F1 | 44.3% | 42.7% | 37.1% |
| Answer coverage | 100.0% | 96.7% | 81.7% |
| First-action accuracy | 33.3% | 33.3% | 30.0% |
| Invalid decision rate | N/A | 3.3% | 18.3% |
| False-answer rate on unanswerable questions | 100.0% | 95.0% | 75.0% |
| Average latency | 0.133 s | 0.343 s | 0.322 s |
| Average retrieval calls | 1.00 | 1.02 | 1.00 |

### Key Findings

- The always-answer baseline achieved the highest overall EM and F1, but it answered every question, including unanswerable ones.
- Neither explicit action prompting nor the sufficiency checklist reliably induced additional retrieval on multi-hop questions.
- C2 reduced the false-answer rate on unanswerable questions relative to C0 and C1, but this came with lower answer coverage and a higher invalid decision rate.
- The controller strategies did not demonstrate a reliable overall advantage over the baseline under the tested configuration.
- The small number of benchmark instances and the specific model and prompting setup limit the generalizability of these findings.

## Repository Structure

```text
rag-agent-context-analysis/
├── data/
│   └── benchmark_60.json
├── src/
│   ├── retriever.py
│   ├── actions.py
│   ├── prompt_budget.py
│   ├── baseline_c0.py
│   ├── run_c0_llm.py
│   ├── evaluate_c0.py
│   ├── c1_agent.py
│   ├── evaluate_c1.py
│   ├── c2_agent.py
│   ├── evaluate_c2.py
│   └── validate_benchmark.py
├── tests/
│   └── test_reliability.py
└── results/
    ├── c0_results_v2.json
    ├── c0_answers_v2.json
    ├── c0_evaluation_v2.json
    ├── c1_results_v2.json
    ├── c1_evaluation_v2.json
    ├── c2_results_v2.json
    └── c2_evaluation_v2.json
```

### Core Components

- **`retriever.py`** — Dense document retrieval using sentence embeddings.
- **`actions.py`** — Parses controller decisions and handles invalid action outputs.
- **`prompt_budget.py`** — Fits prompts within the configured token budget while preserving the question, instructions, and relevant context.
- **`baseline_c0.py` / `run_c0_llm.py`** — Baseline retrieval and answer-generation pipeline.
- **`c1_agent.py`** — Implements explicit action selection and follow-up retrieval.
- **`c2_agent.py`** — Implements checklist-based context-sufficiency assessment.
- **`evaluate_c0.py`, `evaluate_c1.py`, `evaluate_c2.py`** — Evaluate the outputs of each experimental strategy.
- **`validate_benchmark.py`** — Checks benchmark structure and consistency.
- **`tests/test_reliability.py`** — Tests reliability-related behavior.

## Reliability Considerations

The implementation includes safeguards for structured action selection and prompt construction:

- Strict parsing of controller outputs into valid actions.
- A fallback to abstention when an action cannot be parsed.
- Prompt-budget management to keep inputs within the model's configured context limit.
- Benchmark validation and automated reliability tests.
- Versioned v2 experiment outputs to preserve the evaluated results.

## Limitations

- The benchmark contains only 60 controlled instances, limiting statistical power and broad generalization.
- The generator and controllers use the same relatively small instruction-tuned model, `google/flan-t5-base`.
- The benchmark's controlled evidence conditions may not fully reflect production retrieval environments.
- Prompt-based action selection does not guarantee calibrated uncertainty or reliable evidence assessment.
- Results depend on the selected model, retrieval configuration, prompts, and evaluation procedure.
- The observed results do not establish that agentic RAG is inherently worse than baseline RAG; they show that the tested strategies did not demonstrate a reliable advantage in this experimental setup.

## Conclusion

This project evaluates whether lightweight LLM controllers can determine when retrieved evidence is sufficient, when additional retrieval is warranted, and when a question should remain unanswered.

Under the tested configuration, explicit action instructions and a sufficiency checklist did not reliably produce the intended search and abstention behavior. The findings highlight the distinction between generating plausible answers and making reliable evidence-grounded decisions.

Future work could investigate stronger controller models, calibrated confidence estimates, improved evidence verification, larger benchmarks, and evaluation across multiple retrieval and generation configurations.

## References

1. Lewis, P., et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*. NeurIPS.
2. Yao, S., et al. (2023). *ReAct: Synergizing Reasoning and Acting in Language Models*. ICLR.
3. Trivedi, H., et al. (2023). *Interleaving Retrieval with Chain-of-Thought Reasoning for Knowledge-Intensive Multi-Step Questions*. ACL.
4. Jiang, Z., et al. (2023). *Active Retrieval Augmented Generation*. EMNLP.
5. Asai, A., et al. (2024). *Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection*. ICLR.
6. Yang, Z., et al. (2018). *HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering*. EMNLP.
