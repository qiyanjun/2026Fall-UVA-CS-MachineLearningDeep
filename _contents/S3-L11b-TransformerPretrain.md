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
  + BERT's full recipe (Devlin et al., 2019): pretrained on BooksCorpus (800M words) + Wikipedia (2.5B words) with WordPiece tokens, on sentence pairs A, B with masked LM plus next-sentence prediction (IsNext / NotNext, 50/50); Adam with peak LR 1e-4, 10k warm-up steps and linear decay, 1M steps × 256 sequences; fine-tuning with LR 2e-5 to 5e-5, batch 16/32, 2–4 epochs
  + Decoder-only (GPT): the pipeline from base model to chat/reasoning model: (1) pretraining, (2) SFT, (3) preference tuning, (4) RL with verifiable rewards; prompting / in-context learning is not a training stage (it changes the input, not the weights)
  + Each stage through the course's four questions: data, loss, optimization, and what is learned
  + The big picture from "Large Language Models: A Survey" (2024): LLM capabilities (basic, emerging, augmented), a timeline of representative LLMs (GPT, BERT, GPT-2, GPT-3, Codex, InstructGPT, GPT-4, DPO, Claude, Gemini, ...), and how LLMs are built (data cleaning, tokenization, positional encoding, architecture, pretraining, fine-tuning and instruction tuning, alignment, decoding, cost-effective training and adaptation)
- Training tactics
  + A language model is the chain rule: P(x<sub>1</sub>, ..., x<sub>T</sub>) = ∏ P(x<sub>t</sub> \| x<sub>&lt;t</sub>); the output layer is a softmax (multi-class logistic regression) over the vocabulary
  + Tokens: characters, words, or subwords; byte-pair encoding (BPE)
  + Next-token prediction: targets are the inputs shifted by one, teacher forcing, and all T positions trained in parallel thanks to the causal mask
  + Next-token cross-entropy = maximum likelihood; why next-word prediction is implicit multi-task learning
  + Optimizers: why one learning rate is not enough, Adam by hand (momentum + per-weight step size), AdamW (decoupled weight decay)
  + Learning-rate warmup then cosine decay, and a PyTorch training loop with its common bugs (shape errors, training on &lt;pad&gt;, masks applied after the softmax)
- Pretraining and pretrained GPT models
  + Scale: data (Llama-3 saw ∼15 trillion tokens), parameters (0.1B (2018) → 175B (2020) → 405B (2024)), and compute (thousands of GPUs for months)
  + Scaling laws (Kaplan et al., 2020): test loss falls as a power law in compute, data, and parameters; why more parameters don't overfit here (the data grows with the model and is seen about once)
  + GPT-1 → GPT-2 → GPT-3: the same decoder with more of everything; GPT-2 as an unsupervised multitask learner; the rough compute price of GPT-3 175B (~$4.5M)
  + How much data pretraining used in 2024: Llama 3 on about 15T multilingual tokens (vs. 1.8T for Llama 2); DeepSeek-V3 (a 671B mixture-of-experts model, 37B activated per token) on 14.8 trillion tokens
  + How big is 15T tokens? Printed at 1,000 tokens per page and 100 sheets per cm, the stack would be ~1,500 km (1,000 miles) thick
  + Open-source LLMs: what people actually download on Hugging Face (30-day counts as of Oct 31, 2026), led by two embedding models and then GPT-2; not a quality ranking

*(Turning a pretrained base model into an assistant (SFT, LoRA/QLoRA, RLHF, DPO) is covered in "Transformer Post-training".)*


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
7. Besides masked language modeling, what second pretraining task does BERT use, and on what data is it pretrained?
8. How much text were Llama 3 and DeepSeek-V3 pretrained on, and how big is that?
9. Classical bias–variance says more parameters eventually overfit. Why does an LLM's test loss keep falling as parameters grow?

---

## Answer Key

1. **BERT is pretrained as a masked language model: about 15% of tokens are picked (80% replaced by [MASK], 10% by a random token, 10% kept), and the model predicts them from both left and right context.** Unlike GPT, it uses no causal mask, has a loss only at the picked positions, adds a task head for fine-tuning, and does inference in one forward pass with no generation.

2. **The target at position t is the token at position t + 1: the targets are the inputs shifted left by one.** The causal mask stops position t from seeing any token after t, so all T predictions can be computed in parallel from the true prefix (teacher forcing), with no step-by-step loop.

3. **The loss is ℒ(θ) = −(1/N) ∑ log p<sub>θ</sub>(x<sub>t</sub> \| x<sub>&lt;t</sub>), the negative log-likelihood of the text under the chain rule.** Maximizing ∏ p<sub>θ</sub>(x<sub>t</sub> \| x<sub>&lt;t</sub>) is the same as minimizing −∑ log p<sub>θ</sub>(x<sub>t</sub> \| x<sub>&lt;t</sub>), and the labels are free because the next word is already in the text.

4. **Whole words give a huge vocabulary and cannot handle unseen words, while characters give very long sequences.** Subword units keep the vocabulary at a fixed size (about 32K to 200K), split rare words into pieces, and never fall back to an unknown token.

5. **Adam keeps two running averages per weight, the gradient (momentum) and the squared gradient (scale), with a bias fix, so each weight gets its own step size: steep directions are scaled down and flat directions scaled up.** AdamW adds the weight decay λθ directly to the update instead of to the gradient, so it is not divided by √ŝ and every weight decays equally; it is the usual choice for Transformers.

6. **Warmup starts with a small learning rate and increases it linearly over the first steps, while gradients are noisy and Adam's running averages are unsettled; decay (e.g., cosine) takes smaller steps near the end.** Debug with a fixed learning rate first, then add the schedule.

7. **BERT also uses next-sentence prediction: given a sentence pair A, B, it predicts IsNext or NotNext, with the two cases 50/50.** It is pretrained on BooksCorpus (800M words) and Wikipedia (2.5B words) with WordPiece tokens, adding the next-sentence cross-entropy to the masked-LM loss.

8. **Llama 3 was pretrained on about 15T multilingual tokens (vs. 1.8T for Llama 2), and DeepSeek-V3 on 14.8 trillion tokens.** Printed at 1,000 tokens per page and 100 sheets per cm, 15T tokens would make a stack about 1,500 km (1,000 miles) thick.

9. **The training data grows along with the model and is seen only about once, so there is little to memorize.** As a result, test loss keeps falling as a power law in parameters, data, and compute (Kaplan et al., 2020), and the lesson is to scale data and parameters together.

---

## Essay Questions

**Instructions:** Prepare a detailed, essay-format response for each of the following prompts. (Answers not provided).

1. **Compare how an encoder-only model (BERT) and a decoder-only model (GPT) are pretrained.** For each, describe the data, the training targets, the mask, the loss, the optimizer and schedule, and what the pretrained weights are then used for.

2. **Explain why next-token prediction on web-scale text works so well.** Discuss the chain rule and maximum likelihood, why next-word prediction is implicit multi-task learning, and how scale (data, parameters, compute) and scaling laws shaped GPT-1 → GPT-2 → GPT-3 and the models of 2024.

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
| **Next-Sentence Prediction (NSP)** | BERT's second pretraining task: predict whether sentence B follows sentence A (IsNext / NotNext, 50/50). |
| **Scaling Laws** | Test loss falls as a power law in data, parameters, and compute: straight lines on log-log axes. |
