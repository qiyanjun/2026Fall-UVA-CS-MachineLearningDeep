---
LOrder: 335
layout: post
title: Transformer Pretraining
lecture: S3-L19-TransformerTraining
lectureVersion: current
extraContent:
notes: 'Radford et al. (2019), <a href="https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf" target="_blank">Language Models are Unsupervised Multitask Learners (GPT-2)</a> · Kaplan et al. (2020), <a href="https://arxiv.org/abs/2001.08361" target="_blank">Scaling Laws for Neural Language Models</a> · Hoffmann et al. (2022), <a href="https://arxiv.org/abs/2203.15556" target="_blank">Training Compute-Optimal Large Language Models (Chinchilla)</a>'
morenotes: '<a href="https://github.com/karpathy/nanoGPT" target="_blank">nanoGPT</a> + <a href="https://arxiv.org/abs/1711.05101" target="_blank">Loshchilov & Hutter (2019), AdamW</a>'
video:
categories: LLM/Agents
tags:
- 6LLMAgents
- Deep
- Generative
- Training
---

### In this lecture, we cover:
- Recap: the Transformer architecture and its three families (encoder-only, encoder-decoder, decoder-only), from the "Transformer NN Architecture" lecture
- Training stages
  + Why pretraining: layer-wise pretraining on unlabeled data, then supervised fine-tuning (SFT) on labeled data
  + Encoder-only (BERT): masked-LM pretraining (15% of tokens picked: 80% [MASK], 10% random, 10% kept), then fine-tuning with a task head on [CLS] or mean pooling
  + Decoder-only (GPT): the pipeline from base model to chat/reasoning model: (1) pretraining, (2) SFT, (3) preference tuning, (4) RL with verifiable rewards
  + Each stage through the course's four questions: data, loss, optimization, and what is learned
- Training tactics
  + A language model is the chain rule: P(x<sub>1</sub>, ..., x<sub>T</sub>) = ∏ P(x<sub>t</sub> | x<sub>&lt;t</sub>); the output layer is a softmax (multi-class logistic regression) over the vocabulary
  + Tokens: characters, words, or subwords; byte-pair encoding (BPE)
  + Next-token prediction: targets are the inputs shifted by one, teacher forcing, and all T positions trained in parallel thanks to the causal mask
  + Next-token cross-entropy = maximum likelihood; why next-word prediction is implicit multi-task learning
  + Optimizers: why one learning rate is not enough, Adam by hand (momentum + per-weight step size), AdamW (decoupled weight decay)
  + Learning-rate warmup then cosine decay, and a PyTorch training loop with its common bugs (shape errors, training on &lt;pad&gt;, masks applied after the softmax)
  + Counting the parameters of a whole GPT: N × (12D² + 9D) for the layers, plus embeddings and the output layer
- Pretraining and pretrained GPT models
  + Scale (data, parameters, compute) and scaling laws (Kaplan et al., 2020): test loss falls as a power law
  + GPT-1 → GPT-2 → GPT-3: the same decoder with more of everything; GPT-2 as an unsupervised multitask learner; the rough compute price of GPT-3

*(Turning a pretrained base model into an assistant (SFT, LoRA/QLoRA, RLHF, DPO) is covered in the next lecture, "Transformer Post-training".)*


# Study Guide: Transformer Pretraining

---

## Quiz: Short-Answer Questions

**Instructions:** Provide a concise answer (2-3 sentences) for each of the following questions.

1. How is BERT pretrained, and how does its training differ from GPT's?
2. What is the training target for each position of a GPT input sequence, and why can all T positions be trained in one forward pass?
3. Why is pretraining with next-token cross-entropy the same as maximum likelihood?
4. Why do we use subword tokenization (such as BPE) instead of whole words or single characters?
5. How does Adam differ from SGD with a single learning rate, and what does AdamW change?
6. What are learning-rate warmup and decay, and why are they used when training Transformers?

---

## Answer Key

1. **BERT is pretrained as a masked language model: about 15% of tokens are picked (80% replaced by [MASK], 10% by a random token, 10% kept), and the model predicts them from both left and right context.** Unlike GPT, it uses no causal mask, has a loss only at the picked positions, adds a task head for fine-tuning, and does inference in one forward pass with no generation.

2. **The target at position t is the token at position t + 1: the targets are the inputs shifted left by one.** The causal mask stops position t from seeing any token after t, so all T predictions can be computed in parallel from the true prefix (teacher forcing), with no step-by-step loop.

3. **The loss is ℒ(θ) = −(1/N) ∑ log p<sub>θ</sub>(x<sub>t</sub> | x<sub>&lt;t</sub>), the negative log-likelihood of the text under the chain rule.** Maximizing ∏ p<sub>θ</sub>(x<sub>t</sub> | x<sub>&lt;t</sub>) is the same as minimizing −∑ log p<sub>θ</sub>(x<sub>t</sub> | x<sub>&lt;t</sub>), and the labels are free because the next word is already in the text.

4. **Whole words give a huge vocabulary and cannot handle unseen words, while characters give very long sequences.** Subword units keep the vocabulary at a fixed size (about 32K to 200K), split rare words into pieces, and never fall back to an unknown token.

5. **Adam keeps two running averages per weight, the gradient (momentum) and the squared gradient (scale), with a bias fix, so each weight gets its own step size: steep directions are scaled down and flat directions scaled up.** AdamW adds the weight decay λθ directly to the update instead of to the gradient, so it is not divided by √ŝ and every weight decays equally; it is the usual choice for Transformers.

6. **Warmup starts with a small learning rate and increases it linearly over the first steps, while gradients are noisy and Adam's running averages are unsettled; decay (e.g., cosine) takes smaller steps near the end.** Debug with a fixed learning rate first, then add the schedule.

---

## Glossary of Key Terms

| Term | Definition |
|------|------------|
| **Masked Language Model (MLM)** | BERT's pretraining task: predict picked (masked) tokens from their left and right context. |
| **Byte-Pair Encoding (BPE)** | A tokenizer that starts from characters (or bytes) and repeatedly merges the most frequent adjacent pair into a new token. |
| **Next-Token Prediction** | The pretraining task of a GPT: predict token t + 1 from tokens 1, ..., t, trained with cross-entropy (= maximum likelihood). |
| **Teacher Forcing** | Training on the true previous tokens instead of the model's own outputs, so all positions are trained in one pass. |
| **Adam / AdamW** | Gradient descent with momentum and a per-weight step size; AdamW decouples weight decay from the adaptive scaling. |
| **Warmup and Decay** | Increasing the learning rate from near zero over the first steps, then lowering it (e.g., cosine) toward the end. |
| **Scaling Laws** | Test loss falls as a power law in data, parameters, and compute: straight lines on log-log axes. |
