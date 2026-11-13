---
LOrder: 336
layout: post
title: Transformer Post-training
lecture: S3-L19-TransformerTraining
lectureVersion: current
extraContent:
notes: 'Hu et al. (2021), <a href="https://arxiv.org/abs/2106.09685" target="_blank">LoRA: Low-Rank Adaptation of Large Language Models</a>'
morenotes: '<a href="https://arxiv.org/abs/2203.02155" target="_blank">Ouyang et al. (2022), InstructGPT</a> + <a href="https://arxiv.org/abs/2305.18290" target="_blank">Rafailov et al. (2023), DPO</a> + <a href="https://arxiv.org/abs/2305.14314" target="_blank">Dettmers et al. (2023), QLoRA</a>'
notebooks: '<a href="https://colab.research.google.com/github/qiyanjun/2026Fall-UVA-CS-MachineLearningDeep/blob/main/notebook/L19b_DPO.ipynb" target="_blank">notebook/L19b_DPO.ipynb</a> · <a href="https://colab.research.google.com/github/qiyanjun/2026Fall-UVA-CS-MachineLearningDeep/blob/main/notebook/L19b_Reasoning_Models_using_GRPO.ipynb" target="_blank">notebook/L19b_Reasoning_Models_using_GRPO.ipynb</a>'
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
- Recap: the decoder-only training pipeline from base model to chat/reasoning model: (1) pretraining, (2) SFT, (3) preference alignment, (4) RL with verifiable rewards (from "Transformer Pretraining")
  + Each stage through the course's four questions: data, loss, optimization, and what is learned
  + Each stage initializes from the previous stage's parameters: pretraining learns human language, SFT and RLHF (alignment) learn how to respond ("preschool, school, workforce")
- Post-training: from a text continuer to an assistant
  + Why web-pretrained models often don't answer; pretraining supplies the knowledge, SFT and RL make the right choice more likely
  + ChatGPT, "Optimizing Language Models for Dialogue" (OpenAI): no paper, just a blog, released Nov 30 2022; it took 5 days to reach 1M users
- ② Supervised fine-tuning (SFT)
  + Pretrain once, SFT many times: start from the pretrained weights θ<sub>0</sub>, same architecture, softmax head, and loss, a smaller learning rate and a few epochs, on 10<sup>4</sup>–10<sup>6</sup> examples instead of 10<sup>13</sup> pretraining tokens
  + Instruction tuning: (instruction, response) pairs in a multi-turn chat format with roles and special tokens; the model learns when to start (after the &lt;assistant&gt; marker), when to stop (by generating &lt;eos&gt;), and to follow the instruction
  + The SFT loss: the same cross-entropy on the response tokens only, with prompt tokens as context (masked out of the loss sum); "nothing new in the math, only new data"
  + SFT mostly doesn't give LMs new knowledge: what it learns is the format and behavior of an assistant (reference: Zhou et al., LIMA, 2023)
  + SFT splits into two routes: build many specialists (one model per task, e.g., translator, editor) or build one generalist (collect labeled data for all kinds of tasks)
- Parameter-efficient fine-tuning (PEFT)
  + Full fine-tuning is memory-hungry: about 16 bytes per parameter with Adam, ≈ 112 GB for a 7B model before activations, plus a full 14 GB copy for every task
  + PEFT methods (adapters, prefix/prompt tuning, LoRA, QLoRA) freeze the pretrained weights and train typically &lt; 1% new parameters, one small file per task; LoRA is the default today (Hugging Face peft)
  + Adapters (Houlsby et al., 2019): one adapter layer per task, often only about 0.5% to 8% of the model's parameters; useful in multi-task settings
  + LoRA: a low-rank update ΔW = BA to frozen weights (e.g., W<sub>q</sub> and W<sub>v</sub> in every layer), B initialized to 0, merged after training with no extra latency; the rank r is the main hyperparameter; for D = 4096 and r = 8, 65,536 trainable parameters instead of 16,777,216 (0.39%)
  + Quantization: fewer bits per weight (a 7B model takes 28 GB in fp32, 14 GB in fp16, 7 GB in int8, 3.5 GB in int4/NF4); store small integers + one scale per block, dequantize on the fly
  + QLoRA: LoRA adapters (16-bit, trainable) on a 4-bit NF4 frozen model; it fine-tuned a 65B model on a single 48 GB GPU, close to 16-bit full fine-tuning quality (Dettmers et al., 2023); plus paged optimizers and double quantization
  + Catastrophic forgetting ("the surgery succeeded, but the patient died"): LoRA learns less and forgets less
- ③ Preference alignment: RLHF and DPO
  + Human role: SFT needs people to write answers (harder for humans); RLHF only needs their feedback on which answer is better (easier for humans)
  + Stages 1 and 2 put a loss on every token, so the model never sees the answer as a whole; stage 3 scores the whole answer
  + The reward model imitates a human preference judge: trained so that the winning sample y<sub>w</sub> scores higher than the losing sample y<sub>l</sub>, then used as a virtual human to raise the probability of high-scoring answers and lower the probability of low-scoring ones
  + ③ Preference as reinforcement learning (RLHF / GRPO): state s, action a, then a reward r; a policy π(a \| s) or value Q(s, a); maximize the expected total reward with reward modeling and policy gradient; metrics such as win rate and pass@1
  + Does RLHF work? InstructGPT (Ouyang et al., 2022): human ratings rise from GPT to prompted GPT to SFT to PPO-ptx, and a 1.3B PPO-ptx model is already about as preferred as the 175B SFT model
  + DPO: learning directly from preference pairs with one loss, with no reward model and no RL loop (KTO is a related method)
- ④ RL with verifiable rewards: problems plus a checker (answer keys, unit tests) instead of human preference pairs, giving a reasoning model that "thinks" before answering; in the summary, preference/RL training searches over θ with PPO, DPO, or GRPO and yields an aligned or reasoning model
- Summary: the three columns pretrain → SFT → preference/RL, each with its data, task, network, loss, optimizer, and resulting model

*(Generating text from a trained decoder (greedy decoding, sampling, the KV cache, and fast inference) is covered in "Transformer Inference - Decoding and Efficient Inference". Reinforcement learning basics are covered in "Reinforcement Learning and for Post Training LLMs", and RL environments, RLHF, and verifiable rewards in "deep RL Gym".)*


# Study Guide: Transformer Post-training

---

## Quiz: Short-Answer Questions

**Instructions:** Provide a concise answer (2-3 sentences) for each of the following questions.

1. In supervised fine-tuning, why is the loss computed only on the response tokens?
2. "Pretrain once, SFT many times": what stays the same between pretraining and SFT, and what changes?
3. A base model already knows that Mount Everest is the highest peak. What does SFT add?
4. What are the two routes for SFT, and how do they differ?
5. Why can a 24 GB GPU run a 7B model but not fully fine-tune it?
6. How does LoRA reduce the cost of fine-tuning, and why is B initialized to zero?
7. One D × D attention matrix has D = 4096. How many trainable parameters does full fine-tuning need for it, and how many does LoRA with r = 8 need?
8. How does quantization shrink a model, and how does QLoRA combine it with LoRA?
9. Why does RLHF score the whole answer, and how is its reward model trained?
10. How do RLHF and DPO each use human preference pairs?
11. What does stage 4, RL with verifiable rewards, use instead of human preference pairs, and what kind of model does it produce?

---

## Answer Key

1. **The model should learn to produce responses, not to predict the user's prompt.** The prompt tokens are context only (no loss, no gradient), so the SFT loss is the pretraining cross-entropy on the response tokens, which teaches the format and behavior of an assistant.

2. **SFT starts from the pretrained weights θ<sub>0</sub> and keeps the same architecture, the same softmax head, and the same loss function.** What changes is the data and the schedule: 10<sup>4</sup>–10<sup>6</sup> instruction examples instead of 10<sup>13</sup> web tokens, a smaller learning rate, and a few epochs, so the weights move only a little and keep what pretraining learned.

3. **SFT mostly doesn't give the model new knowledge: the knowledge comes from pretraining.** SFT teaches the format and behavior of an assistant, so that "User: Highest peak? AI:" is answered with "Mount Everest [END]" instead of a web-style continuation such as "Haha~ do you know?".

4. **Route 1 builds many specialists: one fine-tuned copy of the pretrained model per task, such as a translator or an editor, each trained only on its own task's data.** Route 2 builds one generalist trained on labeled data for all kinds of tasks, so one model can follow many instructions, even combined ones such as "summarize this article and translate the summary into French".

5. **Running the model needs only the fp16 weights, about 2 bytes per parameter (14 GB).** Full fine-tuning also stores fp16 gradients (14 GB) and Adam's fp32 master weights, momentum, and variance (84 GB), about 16 bytes per parameter or ≈ 112 GB before activations.

6. **LoRA freezes the original weight W<sub>0</sub> and trains a low-rank update ΔW = BA with rank r much smaller than D, so a D × D matrix needs only 2Dr trainable parameters instead of D².** B starts at zero so that ΔW = 0 and training starts exactly from the pretrained model; after training, W = W<sub>0</sub> + ΔW is merged, with no extra latency.

7. **Full fine-tuning trains D² = 4096² = 16,777,216 parameters (about 16M).** LoRA with r = 8 trains D·r + r·D = 2 × 4096 × 8 = 65,536 parameters, only 0.39% of the full matrix.

8. **Quantization stores each weight with fewer bits, as small integers plus one scale per block that are dequantized on the fly: a 7B model drops from 14 GB in fp16 to 3.5 GB in int4.** QLoRA keeps the base model frozen in 4-bit NF4 and trains 16-bit LoRA adapters, with gradients flowing through the 4-bit model into the adapters only; it fine-tuned a 65B model on a single 48 GB GPU.

9. **With a loss on every token, "I will teach you" and "I won't teach you" (to "Teach me evil") differ by only one token, so the model never judges the answer as a whole.** RLHF gives the whole answer a reward; because asking humans every time is expensive, a reward model is trained on preference pairs so that the winning sample y<sub>w</sub> scores higher than the losing sample y<sub>l</sub>, and it then acts as a virtual human judge.

10. **RLHF fits a reward model to human preference pairs ("A is better than B") and then uses RL to raise the reward of the model's answers.** DPO skips the reward model and the RL loop: its loss, −log σ(β[log p<sub>θ</sub>(y<sub>w</sub>\|x)/p<sub>ref</sub>(y<sub>w</sub>\|x) − log p<sub>θ</sub>(y<sub>l</sub>\|x)/p<sub>ref</sub>(y<sub>l</sub>\|x)]), raises the preferred answer and lowers the rejected one relative to the frozen SFT model, and trains like SFT.

11. **It uses problems plus a checker, such as answer keys or unit tests, so the reward can be verified automatically.** Trained with RL (the deck pairs RLHF with GRPO and lists PPO, DPO, and GRPO as the search over θ), it turns the aligned model into a reasoning model that "thinks" before answering, measured with metrics such as pass@1.

---

## Essay Questions

**Instructions:** Prepare a detailed, essay-format response for each of the following prompts. (Answers not provided).

1. **Trace a model through the four training stages (pretraining, SFT, preference alignment, RL with verifiable rewards).** For each stage, give the data, the loss, the optimization, and what is learned, and explain why each stage starts from the previous stage's weights.

2. **You must fine-tune a 7B open model on one consumer GPU.** Explain why full fine-tuning does not fit, and compare adapters, prefix/prompt tuning, LoRA, quantization, and QLoRA in terms of what is trained, memory, and the risk of catastrophic forgetting.

3. **Compare SFT, RLHF, and DPO as ways to shape an assistant's behavior.** Discuss what humans must provide for each, why scoring the whole answer differs from a per-token loss, and what the InstructGPT results suggest about the value of RLHF.

---

## Glossary of Key Terms

| Term | Definition |
|------|------------|
| **Supervised Fine-Tuning (SFT)** | Further training a pretrained model on instruction-response pairs, with the loss only on the response tokens. |
| **Instruction Tuning** | SFT on (instruction, response) pairs in a chat format, turning a text continuer into an assistant. |
| **PEFT (Parameter-Efficient Fine-Tuning)** | Fine-tuning methods (adapters, prefix/prompt tuning, LoRA, QLoRA) that freeze the pretrained weights and train a small number of new parameters. |
| **Adapter** | A small bottleneck MLP block inserted after attention / FFN in each layer; often about 0.5% to 8% of the model's parameters. |
| **LoRA (Low-Rank Adaptation)** | A PEFT method that adds a trainable low-rank update ΔW = BA to frozen weight matrices. |
| **Quantization** | Storing weights with fewer bits (int8, int4/NF4) as small integers plus one scale per block, dequantized on the fly. |
| **QLoRA** | LoRA adapters trained on top of a 4-bit quantized frozen model. |
| **Catastrophic Forgetting** | Losing abilities learned in pretraining when post-training changes the weights too much. |
| **Reward Model** | A model trained on human preference pairs to score whole answers, so that the winning sample y<sub>w</sub> scores higher than the losing sample y<sub>l</sub>. |
| **RLHF** | Reinforcement learning from human feedback: train a reward model on human preferences, then optimize the model with RL. |
| **DPO** | Direct preference optimization: a logistic-regression-style loss on preference pairs, with no reward model and no RL loop. |
| **RL with Verifiable Rewards** | Stage 4 of the pipeline: RL on problems with a checker (answer keys, unit tests), trained with methods such as GRPO, producing a reasoning model. |
