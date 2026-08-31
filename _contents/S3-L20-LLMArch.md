---
LOrder: 330
layout: post
title: LLM Architecture and Training
lecture:
lectureVersion: current
extraContent:
notes: Vaswani et al. (2017), <a href="https://arxiv.org/abs/1706.03762" target="_blank">Attention Is All You Need</a> · Alammar (2018), <a href="https://jalammar.github.io/illustrated-transformer/" target="_blank">The Illustrated Transformer</a>
morenotes: <a href="https://colab.research.google.com/github/qiyanjun/2026Fall-UVA-CS-MachineLearningDeep/blob/main/notebook/L15_Predicting_Movie_Reviews_with_BERT_on_TF_Hub.ipynb">Keras Notebook on BERT for Text </a> + <a href="https://github.com/qiyanjun/2026Fall-UVA-CS-MachineLearningDeep/blob/main/Lectures/S3-25recentLLM-extra.pdf"> [Recent LLM-survey] </a>
video:
categories: LLM/Agents
tags:
- 6LLMAgents
- Deep
- Generative
- Architecture
- Training
---

### In this lecture, we cover:
- Recap: self-attention and the Transformer block (Q/K/V, multi-head attention, residual + layernorm, feed-forward sublayer)
- Decoder-only (GPT-style) vs. encoder-decoder (T5-style) vs. encoder-only (BERT-style) architectures, and why decoder-only dominates modern LLMs
- Tokenization (BPE / SentencePiece) and why vocabulary choice matters
- Positional information: absolute, RoPE, ALiBi, and why long-context models need better position encodings
- The pretraining objective: next-token prediction at scale, and why a simple objective yields broad capability
- Scaling laws (Kaplan et al., Chinchilla): how loss scales with parameters, data, and compute, and what "compute-optimal" training means
- Mixture-of-Experts (MoE) as a way to scale parameter count without proportionally scaling compute
- The three-stage training pipeline overview: pretraining -> supervised fine-tuning (SFT) -> RL post-training (RLHF/RLVR) — see the "deep RL Gym" lecture for the RL post-training stages in depth

# Study Guide: LLM Architecture and Training

## Quiz: Short-Answer Questions

1. Why do most modern LLMs use a decoder-only Transformer architecture rather than the original encoder-decoder design?
2. What problem do RoPE and ALiBi solve that fixed absolute positional embeddings do not?
3. Explain "next-token prediction" as a pretraining objective, and why such a simple objective produces broad downstream capability.
4. What do scaling laws describe, and what is meant by a "compute-optimal" model (Chinchilla-style)?
5. How does a Mixture-of-Experts layer let a model have many more parameters without a proportional increase in inference compute?
6. What are the three broad stages of the modern LLM training pipeline, and what does each stage optimize for?

## Answer Key

1. Decoder-only models are trained with a single, simple causal (left-to-right) next-token objective that works directly for open-ended generation and can be trained on almost any text without needing paired input/output structure, which scales more easily than the encoder-decoder setup originally designed for sequence-to-sequence tasks like translation.
2. Fixed absolute positional embeddings are learned for a specific maximum sequence length and generalize poorly beyond it. RoPE (rotary position embeddings) encodes relative position directly into the attention computation, and ALiBi biases attention scores by distance; both extrapolate better to longer contexts than training-length-bound absolute embeddings.
3. Next-token prediction asks the model to predict the next token given all previous tokens, using cross-entropy loss over the vocabulary. Because natural text encodes an enormous range of facts, reasoning patterns, and styles, a model that gets good at predicting text implicitly learns much of that underlying structure, which transfers to downstream tasks.
4. Scaling laws are empirical power-law relationships between a model's loss and the amount of model parameters, training data, and compute used. A compute-optimal model is one where, for a fixed compute budget, parameters and training tokens are balanced according to these laws to minimize loss, rather than simply making the model as large as possible.
5. An MoE layer replaces a single dense feed-forward block with many parallel "expert" feed-forward blocks and a router that activates only a small subset (e.g., top-2) of experts per token. Total parameter count grows with the number of experts, but per-token compute only grows with the number of *activated* experts, decoupling capacity from inference cost.
6. Pretraining (self-supervised next-token prediction on massive raw text, building general language and world knowledge), Supervised Fine-Tuning / SFT (training on curated instruction-response pairs to make the model follow instructions), and RL post-training (RLHF or RLVR, aligning the model's behavior with human preferences or verifiable task correctness — see the "deep RL Gym" lecture).

## Glossary of Key Terms

| Term | Definition |
|------|------------|
| **Self-Attention** | A mechanism where each token's representation is updated as a weighted combination of all other tokens' representations, with weights computed from Query/Key dot products. |
| **Causal (Masked) Attention** | Self-attention restricted so a token can only attend to itself and earlier tokens, required for autoregressive generation. |
| **Tokenization / BPE** | The process of splitting text into subword units; Byte-Pair Encoding merges frequent character/subword pairs to build a fixed vocabulary. |
| **RoPE (Rotary Position Embedding)** | A relative positional encoding scheme that rotates query/key vectors as a function of position, improving length generalization. |
| **Scaling Laws** | Empirical power-law relationships between model loss and model size, dataset size, and compute. |
| **Chinchilla-optimal** | A compute allocation that balances model size and training tokens (roughly proportionally) to minimize loss for a fixed compute budget. |
| **Mixture-of-Experts (MoE)** | An architecture where a router activates a sparse subset of "expert" sub-networks per token, decoupling parameter count from per-token compute. |
| **Pretraining** | Self-supervised training on large-scale raw text using next-token prediction. |
| **Supervised Fine-Tuning (SFT)** | Fine-tuning a pretrained model on curated instruction/response pairs. |
