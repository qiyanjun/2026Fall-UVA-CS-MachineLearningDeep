---
LOrder: 335
layout: post
title: Transformer NN Training
lecture: S3-L18b-TransformerTrain
lectureVersion: next
extraContent:
notes: Radford et al. (2019), <a href="https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf" target="_blank">Language Models are Unsupervised Multitask Learners (GPT-2)</a> · Kaplan et al. (2020), <a href="https://arxiv.org/abs/2001.08361" target="_blank">Scaling Laws for Neural Language Models</a> · Hoffmann et al. (2022), <a href="https://arxiv.org/abs/2203.15556" target="_blank">Training Compute-Optimal Large Language Models (Chinchilla)</a> · Hu et al. (2021), <a href="https://arxiv.org/abs/2106.09685" target="_blank">LoRA: Low-Rank Adaptation of Large Language Models</a>
morenotes: <a href="https://github.com/karpathy/nanoGPT" target="_blank">nanoGPT</a> + <a href="https://arxiv.org/abs/1904.09751" target="_blank">Holtzman et al. (2019), The Curious Case of Neural Text Degeneration (nucleus sampling)</a> + <a href="https://arxiv.org/abs/2203.02155" target="_blank">Ouyang et al. (2022), InstructGPT</a>
video:
categories: LLM/Agents
tags:
- 6LLMAgents
- Deep
- Generative
- Training
- FineTuning
---


### In this lecture, we cover:
- Recap: the decoder-only Transformer (GPT) from the "Transformer NN Architecture" lecture
  + Masked (causal) self-attention, learned position embeddings, a stack of blocks, and a final linear + softmax over the vocabulary
- From text to training data
  + Tokenization: words, characters, and subword units (byte-pair encoding)
  + Building input/target pairs by shifting the token sequence by one position
- Pretraining a GPT
  + Next-token prediction: cross-entropy loss at every position, averaged over the sequence and the batch
  + Teacher forcing: with the causal mask, all T predictions in a sequence are trained in one forward pass
  + Perplexity as exp(average cross-entropy)
- Making training work
  + AdamW, learning-rate warmup and cosine decay, gradient clipping
  + Dropout, weight tying between the input embedding and the output layer
  + Mini-batches, gradient accumulation, and mixed precision
  + Scaling laws: how loss falls with model size, data, and compute
- Generating text
  + Autoregressive decoding with a KV cache
  + Greedy decoding, temperature, top-k, and top-p (nucleus) sampling
- Post-training: from a pretrained model to an assistant
  + Supervised fine-tuning (SFT) on instruction-response pairs, with the loss only on response tokens
  + Preference tuning: RLHF and DPO (overview; see the extra "Reinforcement Learning" lecture for more)
- Fine-tuning LLMs
  + Full fine-tuning vs. parameter-efficient fine-tuning (PEFT): adapters, prefix/prompt tuning, and LoRA (low-rank update matrices)
  + Decision framework: when to reach for prompting, RAG, or fine-tuning (and when to combine them)


# Study Guide: Transformer NN Training

---

## Quiz: Short-Answer Questions

**Instructions:** Provide a concise answer (2-3 sentences) for each of the following questions.

1. What is the training target for each position of a GPT input sequence?
2. Why can a decoder-only Transformer compute the loss for all T positions in one forward pass?
3. What is perplexity, and how does it relate to the cross-entropy loss?
4. Why do we use subword tokenization (such as BPE) instead of whole words or single characters?
5. What are learning-rate warmup and gradient clipping, and why are they used when training Transformers?
6. How do temperature and top-k sampling change the next-token distribution at generation time?
7. In supervised fine-tuning, why is the loss usually computed only on the response tokens?
8. How does LoRA reduce the cost of fine-tuning compared to updating all model weights?
9. Given a task, what factors would push you toward prompting alone, RAG, or fine-tuning?

---

## Answer Key

1. **The target at position t is the token at position t + 1.** The targets are the input sequence shifted left by one, and the loss is the cross-entropy between the model's predicted distribution and the true next token.

2. **The causal mask stops position t from seeing any token after t, so every position's prediction uses only its own past.** All T predictions can therefore be computed in parallel from the true prefix (teacher forcing), with no step-by-step loop as in an RNN.

3. **Perplexity is exp of the average next-token cross-entropy.** It can be read as the effective number of tokens the model is choosing between; lower is better.

4. **Whole words give a huge vocabulary and cannot handle unseen words, while characters give very long sequences.** Subword units keep the vocabulary at a fixed size (tens of thousands), split rare words into pieces, and keep sequences reasonably short.

5. **Warmup starts training with a small learning rate and increases it over the first steps; gradient clipping rescales gradients whose norm is too large.** Both keep the early updates from being too large, which makes training of deep attention models more stable.

6. **Temperature divides the logits by τ before the softmax: τ < 1 sharpens the distribution and τ > 1 flattens it.** Top-k keeps only the k most likely tokens, renormalizes, and samples from them, which removes the long tail of unlikely tokens.

7. **The model should learn to produce responses, not to predict the user's prompt.** Masking the prompt tokens out of the loss puts all of the training signal on the answer.

8. **LoRA freezes the original weight matrices and adds small trainable low-rank matrices (A and B, with rank r much smaller than the original dimensions) to selected weight matrices.** Only these low-rank matrices are trained, so the number of trainable parameters, and the memory and compute needed for fine-tuning, is much smaller than when updating the full weight matrices.

9. **Prompting alone fits simple tasks with rarely changing data and no training budget; RAG fits tasks that need up-to-date or proprietary knowledge that cannot fit in a prompt or in the model's weights; fine-tuning fits tasks that need consistent formatting or behavior, or a lasting skill or style shift that instructions and examples cannot reliably specify.** These approaches are often combined, for example fine-tuning a model to use retrieved context well.

---

## Glossary of Key Terms

| Term | Definition |
|------|------------|
| **Byte-Pair Encoding (BPE)** | A tokenizer that starts from characters (or bytes) and repeatedly merges the most frequent adjacent pair into a new token. |
| **Next-Token Prediction** | The pretraining task of a GPT: predict token t + 1 from tokens 1, ..., t, trained with cross-entropy. |
| **Teacher Forcing** | Training on the true previous tokens instead of the model's own outputs, so all positions are trained in one pass. |
| **Perplexity** | exp(average cross-entropy per token). |
| **Warmup** | Increasing the learning rate from near zero over the first training steps. |
| **KV Cache** | Storing the keys and values of earlier tokens during generation so each new token needs only one new row of attention. |
| **Temperature** | A scale τ applied to the logits (logits / τ) before the softmax when sampling. |
| **Top-p (Nucleus) Sampling** | Sampling from the smallest set of tokens whose total probability is at least p. |
| **Supervised Fine-Tuning (SFT)** | Further training a pretrained model on instruction-response pairs. |
| **PEFT (Parameter-Efficient Fine-Tuning)** | A family of fine-tuning methods that train a small number of additional parameters instead of the full model. |
| **LoRA (Low-Rank Adaptation)** | A PEFT method that adds trainable low-rank matrices to frozen weight matrices. |
| **RLHF / DPO** | Ways to tune a model toward human preferences: RLHF trains a reward model and uses RL; DPO optimizes directly on preference pairs. |
