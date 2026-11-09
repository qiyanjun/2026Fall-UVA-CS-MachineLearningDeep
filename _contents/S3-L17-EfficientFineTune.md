---
LOrder: 355
layout: post
title: Fine-tuning LLMs
lecture:
extra: true
lectureVersion: current
extraContent:
notes: 'Hu et al. (2021), <a href="https://arxiv.org/abs/2106.09685" target="_blank">LoRA: Low-Rank Adaptation of Large Language Models</a>'
video:
categories: LLM/Agents
tags:
- 6LLMAgents
- Generative
- FineTuning
---

### In this lecture, we cover:
- Fine-tuning approaches: full fine-tuning vs. parameter-efficient fine-tuning (PEFT) — adapters, prefix/prompt tuning, and LoRA (low-rank update matrices)
- Decision framework: when to reach for prompting, RAG, or fine-tuning (and when to combine them)

*(Prompting and RAG are covered in the "Prompting and RAG" lecture.)*

# Study Guide: Fine-tuning LLMs

## Quiz: Short-Answer Questions

1. How does LoRA reduce the cost of fine-tuning compared to updating all model weights?
2. Given a task, what factors would push you toward prompting alone, RAG, or fine-tuning?

## Answer Key

1. LoRA freezes the original weight matrices and injects small trainable low-rank matrices (A and B, with rank r much smaller than the original dimensions) that are added to selected weight matrices. Only these low-rank matrices are trained, so the number of trainable parameters — and the memory/compute needed for fine-tuning — is drastically smaller than updating the full weight matrices.
2. Prompting alone is attractive when the task is simple, data changes rarely, and you cannot afford training infrastructure. RAG is preferred when the task needs up-to-date or proprietary knowledge that cannot fit in a prompt or in the model's weights. Fine-tuning is preferred when you need consistent formatting/behavior, a persistent skill or style shift, or when the task pattern can't be reliably specified through instructions/examples alone — and these approaches are often combined (e.g., fine-tune a model to use retrieved context well).

## Glossary of Key Terms

| Term | Definition |
|------|------------|
| **PEFT (Parameter-Efficient Fine-Tuning)** | A family of fine-tuning methods that train a small number of additional parameters instead of the full model. |
| **LoRA (Low-Rank Adaptation)** | A PEFT method that adds trainable low-rank matrices to frozen weight matrices. |
