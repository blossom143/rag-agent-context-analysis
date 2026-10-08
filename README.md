# Answer, Search, or Abstain: Can RAG Agents Tell When Their Context Is Enough?

A controlled study of whether RAG agents can recognize when their retrieved context is sufficient to answer a question, when they should search again, and when they should abstain.

## Research Question

Can a RAG agent reliably decide whether its current context is sufficient to answer a question, or does it tend to answer even when the available evidence is incomplete?

## Experiment

We construct a 60-example benchmark from HotpotQA with three controlled conditions:

- **Easy (20):** Both supporting documents are available in the initial context.
- **Multi-hop (20):** Only one supporting document is initially available; the second can be found through another retrieval step.
- **Unanswerable (20):** Both original supporting documents are removed from the candidate pool, so additional retrieval cannot recover the answer.

Each example has an oracle action:

- `ANSWER` — the available context is sufficient.
- `SEARCH` — another retrieval step is needed.
- `ABSTAIN` — the answer cannot be supported from the available/candidate evidence.

The benchmark uses a dense retriever based on `BAAI/bge-small-en-v1.5` and evaluates three controllers using `google/flan-t5-base`.

### Controllers

**C0: Always Answer**

Retrieve once and directly generate an answer.

**C1: Action Prompt**

The model explicitly chooses between `ANSWER`, `SEARCH`, and `ABSTAIN`. If it chooses `SEARCH`, a second retrieval step is performed.

**C2: Sufficiency Checklist**

The model uses an explicit checklist to determine whether the retrieved documents contain all information and reasoning steps required to answer.

## Results

| Metric | C0 | C1 | C2 |
|---|---:|---:|---:|
| Easy EM | 60.0% | 60.0% | 60.0% |
| Multi-hop EM | 25.0% | 30.0% | 25.0% |
| Unanswerable EM | 5.0% | 5.0% | 5.0% |
| Multi-hop F1 | 48.8% | 53.8% | 48.8% |
| Action Accuracy | 33.3% | 33.3% | 35.0% |
| Unanswerable False-Answer Rate | 100% | 100% | 100% |

The main finding is that explicit action prompting provides only a small improvement on multi-hop questions. The sufficiency checklist reduces unnecessary searches on easy examples, but neither approach reliably produces `ABSTAIN`.

All three controllers incorrectly attempt to answer every unanswerable example.

## Key Finding

**Retrieval relevance is not the same as evidence sufficiency.**

The retriever can return documents that are semantically related to a question without providing enough evidence to justify an answer. In this experiment, prompting a small pretrained model to reason about context sufficiency did not reliably solve this decision problem.

## Repository Structure

```text
├── data/
│   └── benchmark_60.json
├── src/
│   ├── benchmark.py
│   ├── data.py
│   ├── retriever.py
│   ├── baseline_c0.py
│   ├── run_c0_llm.py
│   ├── c1_agent.py
│   ├── c2_agent.py
│   ├── evaluate_c0.py
│   ├── evaluate_c1.py
│   ├── evaluate_c2.py
│   └── validate_benchmark.py
├── results/
│   ├── c0_answers.json
│   ├── c0_evaluation.json
│   ├── c1_results.json
│   ├── c1_evaluation.json
│   ├── c2_results.json
│   ├── c2_evaluation.json
│   ├── main_results.csv
│   └── figures/
└── report/