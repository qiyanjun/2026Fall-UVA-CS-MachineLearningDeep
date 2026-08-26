---
LOrder: 350
layout: post
title: Prompting, RAG and Fine-tuning
lecture:
lectureVersion: current
extraContent:
notes:
video:
categories: LLM/Agents
tags:
- 6LLMAgents
- Generative
- Prompting
- RAG
- FineTuning
---

### In this lecture, we cover:
- In-context learning: zero-shot, few-shot prompting, and why examples in the prompt can steer behavior without any weight updates
- Prompt engineering patterns: instructions, role/system prompts, chain-of-thought, self-consistency, structured output prompting
- Retrieval-Augmented Generation (RAG): embeddings, vector databases / nearest-neighbor search, chunking strategies, the retrieve-then-generate pipeline
- Why RAG helps with hallucination and stale knowledge, and its limitations (retrieval quality, context-window budget)
- Fine-tuning approaches: full fine-tuning vs. parameter-efficient fine-tuning (PEFT) — adapters, prefix/prompt tuning, and LoRA (low-rank update matrices)
- Decision framework: when to reach for prompting, RAG, or fine-tuning (and when to combine them)

# Study Guide: Prompting, RAG and Fine-tuning

## Quiz: Short-Answer Questions

1. What is "in-context learning," and how does it differ from traditional supervised fine-tuning?
2. What does chain-of-thought prompting do, and why can it improve performance on multi-step reasoning tasks?
3. Describe the retrieve-then-generate pipeline in RAG: what happens at indexing time versus query time?
4. Why does RAG reduce hallucination and help with knowledge that postdates a model's training cutoff?
5. How does LoRA reduce the cost of fine-tuning compared to updating all model weights?
6. Given a task, what factors would push you toward prompting alone, RAG, or fine-tuning?

## Answer Key

1. In-context learning is the ability of a model to adapt its behavior based on examples or instructions placed directly in the prompt at inference time, with no gradient updates to the model's weights. Traditional fine-tuning instead updates the model's parameters using labeled training data.
2. Chain-of-thought prompting asks the model to produce intermediate reasoning steps before its final answer. Spelling out intermediate steps gives the model more "computation" (in the form of generated tokens) to work through a problem, which tends to improve accuracy on arithmetic, logic, and multi-step reasoning tasks compared to asking for the answer directly.
3. At indexing time, a document corpus is split into chunks, each chunk is embedded into a vector, and vectors are stored in a vector index/database. At query time, the user's query is embedded with the same encoder, the nearest chunk vectors are retrieved by similarity search, and the retrieved text is inserted into the prompt as context before the LLM generates its answer.
4. Because the model is given the actual retrieved source text as grounding context, it can quote or paraphrase real information instead of relying purely on facts memorized (and potentially misremembered) during pretraining. Updating the retrieval corpus with new documents also lets the system answer questions about information created after the model's training cutoff, without retraining the model.
5. LoRA freezes the original weight matrices and injects small trainable low-rank matrices (A and B, with rank r much smaller than the original dimensions) that are added to selected weight matrices. Only these low-rank matrices are trained, so the number of trainable parameters — and the memory/compute needed for fine-tuning — is drastically smaller than updating the full weight matrices.
6. Prompting alone is attractive when the task is simple, data changes rarely, and you cannot afford training infrastructure. RAG is preferred when the task needs up-to-date or proprietary knowledge that cannot fit in a prompt or in the model's weights. Fine-tuning is preferred when you need consistent formatting/behavior, a persistent skill or style shift, or when the task pattern can't be reliably specified through instructions/examples alone — and these approaches are often combined (e.g., fine-tune a model to use retrieved context well).

## Glossary of Key Terms

| Term | Definition |
|------|------------|
| **In-Context Learning** | Adapting model behavior via examples/instructions in the prompt, without updating model weights. |
| **Few-Shot Prompting** | Including a handful of input-output examples in the prompt to demonstrate the desired task. |
| **Chain-of-Thought (CoT)** | Prompting the model to generate intermediate reasoning steps before its final answer. |
| **Embedding** | A dense vector representation of text such that semantically similar text has nearby vectors. |
| **Vector Database** | A data store optimized for nearest-neighbor search over embedding vectors. |
| **Retrieval-Augmented Generation (RAG)** | Retrieving relevant text chunks and inserting them into the prompt so the LLM can ground its answer in that context. |
| **Chunking** | Splitting documents into smaller passages for embedding and retrieval. |
| **PEFT (Parameter-Efficient Fine-Tuning)** | A family of fine-tuning methods that train a small number of additional parameters instead of the full model. |
| **LoRA (Low-Rank Adaptation)** | A PEFT method that adds trainable low-rank matrices to frozen weight matrices. |
| **Hallucination** | A confident but factually incorrect or unsupported model output. |
