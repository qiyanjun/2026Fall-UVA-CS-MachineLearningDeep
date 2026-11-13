---
LOrder: 336
layout: post
title: Transformer Post-training
lecture: S3-L19-TransformerTraining
lectureVersion: current
extraContent:
notes: 'Hu et al. (2021), <a href="https://arxiv.org/abs/2106.09685" target="_blank">LoRA: Low-Rank Adaptation of Large Language Models</a>'
morenotes: '<a href="https://arxiv.org/abs/2203.02155" target="_blank">Ouyang et al. (2022), InstructGPT</a> + <a href="https://arxiv.org/abs/2305.18290" target="_blank">Rafailov et al. (2023), DPO</a> + <a href="https://arxiv.org/abs/2305.14314" target="_blank">Dettmers et al. (2023), QLoRA</a>'
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
- Recap: the decoder-only training pipeline from base model to chat/reasoning model: (1) pretraining, (2) SFT, (3) preference tuning, (4) RL with verifiable rewards (from "Transformer Pretraining")
  + Each stage through the course's four questions: data, loss, optimization, and what is learned
- Post-training: from a text continuer to an assistant
  + Why web-pretrained models often don't answer; pretraining supplies the knowledge, SFT and RL make the right choice more likely
  + Instruction tuning: instruction-response pairs, chat format with roles and special tokens, and the SFT loss on response tokens only (loss masking)
  + SFT doesn't give LMs new knowledge (LIMA: less is more for alignment)
  + Full fine-tuning is memory-hungry (≈ 112 GB for a 7B model with Adam); parameter-efficient fine-tuning (PEFT): adapters, prefix/prompt tuning, LoRA, and QLoRA
  + LoRA: a low-rank update ΔW = BA to frozen weights, B initialized to 0, merged after training
  + Quantization (int8 absmax, int4 NF4) and QLoRA: LoRA on a 4-bit frozen model
  + Catastrophic forgetting: LoRA learns less and forgets less
  + RLHF: humans give feedback instead of answers, a reward model as a virtual human judge, and scoring the whole answer instead of every token
  + DPO: learning directly from preference pairs, with no reward model and no RL loop

*(Generating text from a trained decoder (greedy decoding, sampling, the KV cache, and fast inference) is covered in the next lecture, "Transformer Inference - Decoding and Efficient Inference". For RL post-training (policy gradient, PPO, GRPO) in more depth, see the extra "Reinforcement Learning" lecture.)*


# Study Guide: Transformer Post-training

---

## Quiz: Short-Answer Questions

**Instructions:** Provide a concise answer (2-3 sentences) for each of the following questions.

1. In supervised fine-tuning, why is the loss computed only on the response tokens?
2. How does LoRA reduce the cost of fine-tuning, and why is B initialized to zero?
3. A GPT with D = 768 and 12 layers uses LoRA with r = 4 on W<sub>q</sub> and W<sub>v</sub> in every layer. How many trainable parameters does LoRA add?
4. How do RLHF and DPO each use human preference pairs?
5. Given a task, what factors would push you toward prompting alone, RAG, or fine-tuning?

---

## Answer Key

1. **The model should learn to produce responses, not to predict the user's prompt.** The prompt tokens are context only (no loss, no gradient), so the SFT loss is the pretraining cross-entropy on the response tokens, which teaches the format and behavior of an assistant.

2. **LoRA freezes the original weight W<sub>0</sub> and trains a low-rank update ΔW = BA with rank r much smaller than D, so a D × D matrix needs only 2Dr trainable parameters instead of D².** B starts at zero so that ΔW = 0 and training starts exactly from the pretrained model; after training, W = W<sub>0</sub> + ΔW is merged, with no extra latency.

3. **Each matrix adds D·r + r·D = 2 × 768 × 4 = 6,144 parameters.** With 2 matrices in each of 12 layers, the total is 24 × 6,144 = 147,456, about 1% of one 768 × 768 matrix per adapted matrix.

4. **RLHF fits a reward model to human preference pairs ("A is better than B") and then uses RL to raise the reward of the model's answers.** DPO skips the reward model and the RL loop: its loss, −log σ(β[log p<sub>θ</sub>(y<sub>w</sub>|x)/p<sub>ref</sub>(y<sub>w</sub>|x) − log p<sub>θ</sub>(y<sub>l</sub>|x)/p<sub>ref</sub>(y<sub>l</sub>|x)]), raises the preferred answer and lowers the rejected one relative to the frozen SFT model, and trains like SFT.

5. **Prompting alone fits simple tasks with rarely changing data and no training budget; RAG fits tasks that need up-to-date or proprietary knowledge; fine-tuning fits tasks that need consistent formatting or behavior, or a lasting skill or style shift.** These approaches are often combined, for example fine-tuning a model to use retrieved context well.

---

## Glossary of Key Terms

| Term | Definition |
|------|------------|
| **Supervised Fine-Tuning (SFT)** | Further training a pretrained model on instruction-response pairs, with the loss only on the response tokens. |
| **PEFT (Parameter-Efficient Fine-Tuning)** | Fine-tuning methods (adapters, prefix/prompt tuning, LoRA) that freeze the pretrained weights and train a small number of new parameters. |
| **LoRA (Low-Rank Adaptation)** | A PEFT method that adds a trainable low-rank update ΔW = BA to frozen weight matrices. |
| **Quantization** | Storing weights with fewer bits (int8, int4/NF4) plus a scale per block, dequantized on the fly. |
| **QLoRA** | LoRA adapters trained on top of a 4-bit quantized frozen model. |
| **Catastrophic Forgetting** | Losing abilities learned in pretraining when post-training changes the weights too much. |
| **RLHF** | Reinforcement learning from human feedback: train a reward model on human preferences, then optimize the model with RL. |
| **DPO** | Direct preference optimization: a logistic-regression-style loss on preference pairs, with no reward model and no RL loop. |
