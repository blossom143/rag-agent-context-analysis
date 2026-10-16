# Section 4 (Method) Analysis & Step-by-Step Workflow Diagrams

This document breaks down **Section 4: Method** of the research draft *"Answer, Search, or Abstain: Can RAG Agents Tell When Their Context Is Enough?"* into structured architectural diagrams and step-by-step technical workflows.

---

## 1. High-Level Overview of Section 4

Section 4 defines a controlled experimental methodology designed to isolate and evaluate how RAG agents make the critical decision of whether their retrieved context is sufficient. It is organized into four core pillars:

1. **Section 4.1 (Benchmark):** Construction of a 900-question testbed spanning three controlled conditions (Easy, Multi-hop, Unanswerable) with query-time corpus filtering.
2. **Section 4.2 (Oracle Labels):** A formal mathematical definition of the ground-truth optimal action at every agent step ($a_t^* \in \{\text{ANSWER}, \text{SEARCH}, \text{ABSTAIN}\}$).
3. **Section 4.3 (System):** The unified multi-step agent runtime loop (retrieve $\to$ rerank $\to$ controller $\to$ branch) with controlled fixed components.
4. **Section 4.4 (Controllers):** An ablation hierarchy of five decision controllers (C0–C4) ranging from naive single-shot to prompted reasoning, checklist verification, signal classification, and a LoRA policy adapter.

---

## 2. Step-by-Step Workflow Diagrams

### Diagram 1: The Agent Execution Loop (Section 4.3)
Visualizes the multi-step runtime loop ($t \in \{1, 2, 3\}$), showing how retrieved passages are processed and branched by the controller.

```mermaid
flowchart TD
    Start(["User Question q"]) --> Init["Initialize: Step t = 1<br/>Cumulative Context R_0 = ∅"]
    Init --> QueryPrep["Prepare Search Query q_t<br/>(Initial: q_1 = q)"]
    
    subgraph S1 ["Step 1: Dense Retrieval"]
        QueryPrep --> Embed["BGE Embedding Lookup"]
        Embed --> FAISS[("Unified FAISS Index<br/>50k-100k Paragraphs")]
        FAISS --> Top20["Retrieve Top-20 Passages"]
    end

    subgraph S2 ["Step 2: Cross-Encoder Reranking"]
        Top20 --> Reranker["BGE Cross-Encoder Reranker"]
        Reranker --> Top5["Select Top-5 Reranked Passages"]
        Top5 --> UpdateContext["Update Cumulative Context:<br/>R_t = R_(t-1) ∪ Top-5"]
    end

    subgraph S3 ["Step 3: Controller Decision"]
        UpdateContext --> Controller{"Decision Controller<br/>(C0, C1, C2, C3, or C4)"}
    end

    subgraph S4 ["Step 4: Branch & Action Execution"]
        Controller -->|"Action = ANSWER"| AnswerGen["Fixed Answer Generator<br/>(Generates Final Grounded Answer)"]
        Controller -->|"Action = ABSTAIN"| AbstainOut["Fixed Output:<br/>'Information Not Found'"]
        Controller -->|"Action = SEARCH"| CheckLimit{"Step Budget Check:<br/>t < 3?"}
        
        CheckLimit -->|"Yes (t < 3)"| Rewriter["Query Rewriter<br/>(Produces q_(t+1))"]
        Rewriter --> IncStep["Increment Step: t = t + 1"]
        IncStep --> QueryPrep
        
        CheckLimit -->|"No (t = 3 reached)"| BudgetFallback["Budget Exhausted<br/>Fallback: Fixed Abstain / Termination"]
    end

    AnswerGen --> Finish(["End of Episode"])
    AbstainOut --> Finish
    BudgetFallback --> Finish

    classDef proc fill:#e1f5fe,stroke:#0288d1,stroke-width:1.5px;
    classDef decision fill:#fff9c4,stroke:#fbc02d,stroke-width:1.5px;
    classDef term fill:#e8f5e9,stroke:#388e3c,stroke-width:1.5px;
    classDef storage fill:#ede7f6,stroke:#512da8,stroke-width:1.5px;

    class Init,QueryPrep,Embed,Top20,Reranker,Top5,UpdateContext,Rewriter,IncStep proc;
    class Controller,CheckLimit decision;
    class Start,Finish,AnswerGen,AbstainOut,BudgetFallback term;
    class FAISS storage;
```

---

### Diagram 2: Benchmark Construction & Query-Time Filtering (Section 4.1)
Shows how 900 evaluation instances across three conditions are derived from HotpotQA using query-time filtering without rebuilding indices.

```mermaid
flowchart TD
    subgraph DataPrep ["Data Assembly"]
        HP["HotpotQA Dev Set<br/>(Question + 2 Gold Passages G)"] --> FilterPool["Select Evaluation Questions<br/>(900 Total)"]
        HP --> CorpusBuild["Extract Supporting + Distractor Paragraphs"]
        CorpusBuild --> Indexing["Embed with BGE Small"]
        Indexing --> FAISSIndex[("Single Static FAISS Index<br/>(50k - 100k Paragraphs)")]
    end

    subgraph Conditions ["Three Controlled Benchmark Conditions (300 Questions Each)"]
        FilterPool --> CondEasy["1. Easy (300 Qs)<br/>Standard Retrieval"]
        FilterPool --> CondMulti["2. Multi-hop (300 Qs)<br/>1st hop partial retrieval"]
        FilterPool --> CondUnans["3. Unanswerable (300 Qs)<br/>Gold Passages Filtered"]
    end

    subgraph QueryExecution ["Query-Time Masking Mechanism"]
        CondEasy -->|"Search index normally"| RetEasy["Initial Retrieval returns<br/>both gold passages (G ⊆ R_1)"]
        CondMulti -->|"Search index normally"| RetMulti["1st Retrieval returns only 1st hop.<br/>2nd hop requires rewrites (t > 1)"]
        CondUnans -->|"Query-time ID filter:<br/>Exclude G from candidate pool"| RetUnans["Gold passages blocked at query time.<br/>Retriever returns distractors only."]
    end

    classDef box fill:#f3e5f5,stroke:#7b1fa2,stroke-width:1.5px;
    classDef cond fill:#e0f2f1,stroke:#00796b,stroke-width:1.5px;
    classDef ret fill:#e8eaf6,stroke:#303f9f,stroke-width:1.5px;

    class HP,FilterPool,CorpusBuild,Indexing,FAISSIndex box;
    class CondEasy,CondMulti,CondUnans cond;
    class RetEasy,RetMulti,RetUnans ret;
```

---

### Diagram 3: Step-Level Oracle Action Derivation Logic (Section 4.2)
Illustrates the exact decision rule that generates ground-truth supervision and evaluation labels at every step $t$.

```mermaid
flowchart TD
    Start(["Evaluate Step t for Question q"]) --> CheckCorpus{"Are Gold Passages G in the Searchable Corpus?<br/>(G ⊆ corpus?)"}
    
    CheckCorpus -->|"No (Gold removed/filtered)"| ActionAbstain["Oracle Action a_t* = ABSTAIN<br/>(Evidence is fundamentally unreachable)"]
    
    CheckCorpus -->|"Yes (Gold is available)"| CheckRetrieved{"Are All Gold Passages Retrieved?<br/>(G ⊆ R_t?)"}
    
    CheckRetrieved -->|"Yes (All gold found)"| ActionAnswer["Oracle Action a_t* = ANSWER<br/>(Context is sufficient to answer)"]
    CheckRetrieved -->|"No (Partial or no gold found)"| ActionSearch["Oracle Action a_t* = SEARCH<br/>(More evidence needed; search again)"]

    ActionAnswer --> Outcome(["Assign Step Label a_t*"])
    ActionSearch --> Outcome
    ActionAbstain --> Outcome

    classDef startNode fill:#eceff1,stroke:#455a64,stroke-width:1.5px;
    classDef check fill:#fff3e0,stroke:#e65100,stroke-width:1.5px;
    classDef answer fill:#c8e6c9,stroke:#2e7d32,stroke-width:1.5px;
    classDef search fill:#bbdefb,stroke:#1565c0,stroke-width:1.5px;
    classDef abstain fill:#ffcdd2,stroke:#c62828,stroke-width:1.5px;

    class Start,Outcome startNode;
    class CheckCorpus,CheckRetrieved check;
    class ActionAnswer answer;
    class ActionSearch search;
    class ActionAbstain abstain;
```

---

### Diagram 4: Controller Architectures & Decision Pathways (Section 4.4)
Compares how each of the five controllers processes the cumulative retrieved passages $R_t$ to produce a decision.

```mermaid
flowchart TD
    InputContext["Input: Question q + Retrieved Top-5 Passages R_t"]

    subgraph C0 ["C0: Single-Shot RAG (Baseline)"]
        InputContext --> C0_Action["Directly invoke Answer Generator<br/>(Always ANSWER at t=1, no SEARCH/ABSTAIN)"]
    end

    subgraph C1 ["C1: ReAct Prompt"]
        InputContext --> C1_Prompt["ReAct Prompting:<br/>Reason briefly about sufficiency"]
        C1_Prompt --> C1_Constrained["Constrained Decoding<br/>Forces single valid token: [ANSWER / SEARCH / ABSTAIN]"]
    end

    subgraph C2 ["C2: Sufficiency Checklist"]
        InputContext --> C2_List["Structured Extraction:<br/>List required facts from q"]
        C2_List --> C2_Verify["Fact Verification Matrix:<br/>Mark each fact FOUND [with passage ID] or MISSING"]
        C2_Verify --> C2_Rule{"Deterministic Rule:<br/>Are all facts verified?"}
        C2_Rule -->|"All Found"| C2_Ans["ANSWER"]
        C2_Rule -->|"Facts Missing"| C2_Search["SEARCH / ABSTAIN"]
    end

    subgraph C3 ["C3: Signal Classifier (Lightweight)"]
        InputContext --> C3_Features["Extract 5 Fast Signals:<br/>1. Reranker scores<br/>2. Top-2 score sum<br/>3. Passage novelty (new vs seen)<br/>4. Step number t<br/>5. Draft answer confidence"]
        C3_Features --> C3_LR["Multiclass Logistic Regression"]
        C3_LR --> C3_Out["Predicted Action: [ANSWER / SEARCH / ABSTAIN]"]
    end

    subgraph C4 ["C4: LoRA Decision Policy"]
        InputContext --> C4_Model["Fine-Tuned 3B LLM (4-bit)<br/>Trained on Oracle Labels"]
        C4_Model --> C4_Token["Direct Next-Token Prediction:<br/>Outputs single action token"]
    end

    classDef inNode fill:#f5f5f5,stroke:#616161,stroke-width:1.5px;
    classDef c0 fill:#efebe9,stroke:#4e342e,stroke-width:1.2px;
    classDef c1 fill:#e3f2fd,stroke:#1565c0,stroke-width:1.2px;
    classDef c2 fill:#e8f5e9,stroke:#2e7d32,stroke-width:1.2px;
    classDef c3 fill:#fff8e1,stroke:#f57f17,stroke-width:1.2px;
    classDef c4 fill:#f3e5f5,stroke:#6a1b9a,stroke-width:1.2px;

    class InputContext inNode;
    class C0_Action c0;
    class C1_Prompt,C1_Constrained c1;
    class C2_List,C2_Verify,C2_Rule,C2_Ans,C2_Search c2;
    class C3_Features,C3_LR,C3_Out c3;
    class C4_Model,C4_Token c4;
```

---

### Diagram 5: Training Pipeline for Learned Controllers C3 & C4 (Section 4.4 & 5)
Visualizes the offline rollout and data preparation pipeline that trains C3 and C4 on oracle step labels without test data contamination.

```mermaid
flowchart TD
    TrainQ["~3,000 HotpotQA Training Questions<br/>(Disjoint from 900 Test Qs)"] --> AgentRollout["Agent Step Simulation<br/>(Rollout agent loop with dense retrieval)"]
    
    AgentRollout --> StepCapture["Capture Agent State at each step t:<br/>(q, R_t, retrieval signals)"]
    
    OracleEngine["Oracle Action Engine (Section 4.2)"] --> AssignLabel["Derive Ground Truth a_t*:<br/>ANSWER / SEARCH / ABSTAIN"]
    StepCapture --> AssignLabel
    
    AssignLabel --> RawDataset["Step-Level Dataset<br/>(~5,000 - 8,000 Labeled Step Examples)"]
    
    subgraph Balancing ["Class Balancing & Data Splitting"]
        RawDataset --> Balance["Resampling to balance action distribution<br/>(Compensate for skewed frequency)"]
        Balance --> Split["90% Train Set / 10% Validation Holdout"]
    end

    subgraph TrainC3 ["Train Controller C3"]
        Split --> FeatExtract["Extract 5 Numerical Signal Features"]
        FeatExtract --> LogReg["Fit Multiclass Logistic Regression<br/>(Scikit-Learn)"]
    end

    subgraph TrainC4 ["Train Controller C4"]
        Split --> TokenFormat["Format Prompt + Target Action Token"]
        TokenFormat --> LoRATrain["Fine-tune 3B Base Model (4-bit QLoRA)<br/>Objective: Cross-Entropy on Single Action Token"]
    end

    classDef source fill:#e0f7fa,stroke:#00838f,stroke-width:1.5px;
    classDef proc fill:#f9fbe7,stroke:#827717,stroke-width:1.5px;
    classDef pipe fill:#e1f5fe,stroke:#0277bd,stroke-width:1.5px;
    classDef mod fill:#ede7f6,stroke:#4527a0,stroke-width:1.5px;

    class TrainQ,OracleEngine source;
    class AgentRollout,StepCapture,AssignLabel,RawDataset proc;
    class Balance,Split pipe;
    class FeatExtract,LogReg,TokenFormat,LoRATrain mod;
```

---

## 3. Summary of Key Methodological Safeguards

| Component | Methodological Safeguard in Draft | Impact on Rigor |
| :--- | :--- | :--- |
| **Controlled Variates** | Retriever (`BGE-small`), reranker (`BGE-reranker`), and answer prompt are identical across all controllers. | Isolates controller decision performance from retriever quality. |
| **Query-Time Masking** | Evaluates unanswerable questions by filtering gold passages at retrieval time. | Keeps a single FAISS index without rebuilding for each condition. |
| **Exact Oracle** | Ground truth $a_t^*$ is mathematically grounded in gold passage set membership ($G \subseteq R_t$). | Eliminates subjective annotation; enables granular per-step confusion matrices. |
| **Data Hygiene** | 3k training questions and 900 dev test questions have strictly zero overlap. | Prevents memorization of HotpotQA entity pairs. |
| **Class Balancing** | Step-level dataset (5k–8k steps) is resampled across `ANSWER`, `SEARCH`, `ABSTAIN`. | Prevents trivial majority-class collapse towards `ANSWER`. |
