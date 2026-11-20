---
LOrder: 350
layout: post
title: "Adapting LLMs: Prompting, Test-time Scaling, RAG and Agents"
lecture: S3-L21-Prompting-RAG-TTS
lectureVersion: current
extraContent:
notes: 'Wei et al. (2022), <a href="https://arxiv.org/abs/2201.11903" target="_blank">Chain-of-Thought Prompting Elicits Reasoning in Large Language Models</a> · Lewis et al. (2020), <a href="https://arxiv.org/abs/2005.11401" target="_blank">Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks</a>'
morenotes: '<a href="https://arxiv.org/abs/2203.11171" target="_blank">Wang et al. (2023), Self-Consistency</a> + <a href="https://arxiv.org/abs/2408.03314" target="_blank">Snell et al. (2025), Scaling Test-Time Compute Optimally</a> + <a href="https://arxiv.org/abs/2501.19393" target="_blank">Muennighoff et al. (2025), s1: Simple Test-Time Scaling</a> + <a href="https://arxiv.org/abs/2307.03172" target="_blank">Liu et al. (2024), Lost in the Middle</a> + <a href="https://arxiv.org/abs/2309.15217" target="_blank">Es et al. (2024), RAGAS</a> + <a href="https://github.com/langchain-ai/rag-from-scratch" target="_blank">RAG from scratch (code)</a>'
notebooks: '<a href="https://colab.research.google.com/github/qiyanjun/2026Fall-UVA-CS-MachineLearningDeep/blob/main/notebook/L21-PromptRAG-notebook-prompting-retrieval-rag.ipynb" target="_blank">notebook/L21-PromptRAG-notebook-prompting-retrieval-rag.ipynb</a>'
video:
categories: LLM/Agents
tags:
- 6LLMAgents
- Generative
- Prompting
- RAG
---

### In this lecture, we cover:
- Two ways to improve a frozen LLM: change what goes in (prompting, in-context learning, RAG), or spend more compute per question (test-time scaling)
- Prompting and in-context learning
  + Anatomy of a prompt: system/instruction, optional context, optional examples, question
  + Zero-shot vs. few-shot prompting; GPT-3 few-shot generalization
  + In-context learning is not training: no gradient step, and its known caveats (sensitivity to wording, order, label balance)
  + Emergent abilities of large language models
  + Chain-of-thought prompting (few-shot and zero-shot "Let's think step by step"), and why magic phrases fade as models improve
  + A practical prompting checklist: prompts are hyperparameters, so tune them on dev data
- Test-time scaling (TTS): more compute per question, paid on every query
  + Four ways: parallel (sample N, vote), sequential (draft, revise, think longer), hybrid (search a tree of steps), internal (reasoning models trained to think longer)
  + Self-consistency and majority voting; coverage (pass@k = 1 − (1 − p)^k) vs. selection
  + Best-of-N with a verifier: programs, outcome reward models (ORM), process reward models (PRM)
  + Tree of Thoughts; verifiers + search (best-of-N, beam, lookahead) at a fixed budget
  + Reasoning models, budget forcing (s1), controlling reasoning length (L1), and compute-optimal test-time scaling
- Retrieval-Augmented Generation (RAG): a seven-step pipeline
  + Offline: ingest and chunk (overlapping windows), embed (sparse TF-IDF/BM25 vs. dense embeddings), index (nearest-neighbor search)
  + Online: hybrid retrieval with reciprocal rank fusion (RRF), cross-encoder reranking, generating with citations
  + RAG vs. long context; agentic RAG, GraphRAG, adaptive routing; where knowledge lives (weights, context, store)
  + Evaluating RAG with the RAG triad: context relevance (recall@k, hit@k), answer correctness (exact match, token F1), groundedness (faithfulness)
  + Lost in the middle, common RAG failure types, and frameworks to use
- Tools, agents and context engineering
  + Tool use as next-token prediction: search, code (Program of Thought), plug-ins, computer use
  + AI agents: goal, action, observation; multi-agent debate and role play
  + Context engineering: select what's needed (RAG, rerank, pruning, tool RAG, memory RAG), compress old history, split across agents
- Choosing among prompt, think, retrieve, act, or fine-tune

*(Fine-tuning, including LoRA and QLoRA, is covered in the "Transformer Post-training" lecture; agents are covered in more depth in the "Agents and Tool Use" lecture.)*

# Study Guide: Adapting LLMs: Prompting, Test-time Scaling, RAG and Agents

## Quiz: Short-Answer Questions

1. What is "in-context learning," and how does it differ from traditional supervised fine-tuning?
2. What does chain-of-thought prompting do, and why can it improve performance on multi-step reasoning tasks?
3. Name the four ways to scale test-time compute, and say how test-time scaling differs from pretraining scaling in when its cost is paid.
4. Each sample from a model is right with probability p = 0.1. What is the coverage (pass@k) with k = 10 samples, and why is coverage only an upper bound?
5. When does majority voting over N samples help, and when does it hurt?
6. Describe the RAG pipeline: what happens offline (indexing time) versus online (for every question)?
7. Sparse (TF-IDF, BM25) and dense (embedding) retrieval each fail differently. Give one failure of each, and say why RAG systems use both.
8. A prediction is "Logistic regression minimizes the log loss." and the reference is "It minimizes the cross entropy loss." What are the exact match and token F1 scores, and what does this show about token F1?
9. What does the faithfulness (groundedness) metric measure, and why does it not need a reference answer?
10. What is context engineering, and name two ways to keep an agent's context small.

## Answer Key

1. In-context learning is the ability of a model to adapt its behavior based on examples or instructions placed directly in the prompt at inference time, with no gradient updates to the model's weights. The effect ends with the call, so examples must be resent on the next call. Traditional fine-tuning instead updates the model's parameters using labeled training data.
2. Chain-of-thought prompting asks the model to produce intermediate reasoning steps before its final answer, either by showing worked examples (few-shot CoT) or with a phrase such as "Let's think step by step" (zero-shot CoT). Intermediate steps mean more tokens, and so more computation, which helps multi-step problems such as math and logic, but not simple lookups.
3. Parallel (sample N answers and vote or pick one), sequential (think longer, or draft then revise), hybrid (search a tree of partial steps, with a verifier keeping good ones), and internal (a reasoning model trained to decide how long to think). Pretraining scaling (bigger model, more data) is paid once; test-time scaling is paid on every query. Only the inference budget changes.
4. Coverage = 1 − (1 − 0.1)^10 = 1 − 0.9^10 ≈ 0.65. It is an upper bound because it only asks whether at least one sample is right; a selection step (majority vote or a verifier) must still pick that right answer, and selection decides how much of the coverage you keep.
5. Voting sharpens the answer the model usually gives: it helps when each sample is right with p > 0.5 (for p = 0.6, accuracy rises from 0.60 with 1 sample to 0.85 with 25) and hurts when p < 0.5 (for p = 0.4, it falls from 0.40 to 0.15). Correlated samples give smaller real gains.
6. Offline, the index is built once: documents are ingested and chunked into overlapping windows (step 1), each chunk is embedded as a vector (step 2), and the vectors are indexed for fast nearest-neighbor search (step 3). Online, for every question: hybrid retrieval (step 4), reranking (step 5), and generating an answer with citations (step 6). Step 7 evaluates retrieval quality and answer faithfulness, then fixes the weakest step.
7. Sparse retrieval matches exact shared words, so it misses synonyms ("log loss" vs. "cross-entropy"). Dense retrieval matches meaning but is weak on names, codes and numbers (such as sklearn's C). Because they fail differently, hybrid retrieval combines both, for example with reciprocal rank fusion.
8. Exact match = 0, because the strings differ. After normalizing, the prediction tokens are {logistic, regression, minimizes, log, loss} and the reference tokens are {it, minimizes, cross, entropy, loss}; they share 2 tokens, so P = R = 2/5 and F1 = 0.40. Token F1 rewards shared words, not meaning: it does not know that "log loss" equals "cross entropy".
9. Faithfulness splits the answer into claims and checks each against the retrieved sources: faithfulness = supported claims / all claims (for example, 2 of 3 claims supported gives ≈ 0.67). It compares the answer with the retrieved context rather than with a reference answer, so no reference is needed; usually an LLM judges, which is cheap and scalable but can also err.
10. Context engineering means putting into the model's context only what is needed and clearing out what isn't. Ways to do this include selecting (RAG over documents, search then rerank, pruning sentences, tool RAG, memory RAG), compressing (summarizing old history), and splitting the work across multiple agents so that each agent has its own context.

## Glossary of Key Terms

| Term | Definition |
|------|------------|
| **In-Context Learning** | Adapting model behavior via examples/instructions in the prompt, without updating model weights. |
| **Few-Shot Prompting** | Including a handful of input-output examples in the prompt to demonstrate the desired task. |
| **Emergent Ability** | An ability that is not present in smaller models but is present in larger models. |
| **Chain-of-Thought (CoT)** | Prompting the model to generate intermediate reasoning steps before its final answer. |
| **Test-Time Scaling (TTS)** | Spending more inference compute per question (sampling, revising, searching, thinking longer) with the weights unchanged. |
| **Self-Consistency** | Sampling several chain-of-thought answers and returning the majority final answer. |
| **Coverage (pass@k)** | The probability that at least one of k samples is correct. |
| **Best-of-N** | Sampling N answers, scoring each with a verifier, and returning the best. |
| **Process Reward Model (PRM)** | A verifier that scores every reasoning step, in contrast to an outcome reward model (ORM) that scores only the final answer. |
| **Budget Forcing** | Controlling reasoning length at test time: stopping at a budget to force an answer, or appending "Wait" to keep thinking (s1). |
| **Embedding** | A dense vector representation of text such that semantically similar text has nearby vectors. |
| **TF-IDF / BM25** | Sparse retrieval scores with one dimension per word; they match exact shared words. |
| **Vector Database** | A data store optimized for nearest-neighbor search over embedding vectors. |
| **Retrieval-Augmented Generation (RAG)** | Retrieving relevant text chunks and inserting them into the prompt so the LLM can ground its answer in that context. |
| **Chunking** | Splitting documents into smaller, overlapping passages for embedding and retrieval. |
| **Reciprocal Rank Fusion (RRF)** | Merging ranked lists (e.g., BM25 and dense) using only each item's rank. |
| **Reranker (Cross-Encoder)** | A model that encodes the question and a chunk together to score relevance more accurately than separate embeddings. |
| **Recall@k / Hit@k** | The fraction of relevant chunks found in the top-k / whether any relevant chunk is in the top-k. |
| **Token F1** | A SQuAD answer score comparing bags of normalized words between prediction and reference. |
| **Faithfulness (Groundedness)** | The fraction of an answer's claims supported by the retrieved sources. |
| **Lost in the Middle** | The tendency of LLMs to use information at the start and end of a long context better than in the middle. |
| **AI Agent** | An LLM that pursues a goal by taking actions with tools and reading back observations, in a loop. |
| **Context Engineering** | Selecting, compressing and splitting what goes into the model's context so that only what's needed is kept. |
| **Hallucination** | A confident but factually incorrect or unsupported model output. |
