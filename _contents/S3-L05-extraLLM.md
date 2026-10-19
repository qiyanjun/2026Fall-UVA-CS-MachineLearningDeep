---
LOrder: 220
layout: post
title: Overview of GenAI model training  
lecture: S3_Peng_LLM_train
extraContent: 
notes: 'Ouyang et al. (2022), <a href="https://arxiv.org/abs/2203.02155" target="_blank">Training Language Models to Follow Instructions with Human Feedback (InstructGPT)</a> · Rafailov et al. (2023), <a href="https://arxiv.org/abs/2305.18290" target="_blank">Direct Preference Optimization: Your Language Model is Secretly a Reward Model</a> · DeepSeek-AI (2025), <a href="https://arxiv.org/abs/2501.12948" target="_blank">DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning</a>'
morenotes: '<a href="https://colab.research.google.com/drive/1i7aDdW-RHNNPgCCHW9Ky858riqYbrN1R" target="_blank">DPO Colab Notebook</a> + <a href="https://colab.research.google.com/drive/1hydSFeV8hpTleZTp5EwpoDCjrqRp8JQ7" target="_blank">GRPO Colab Notebook</a> + <a href="https://huggingface.co/docs/trl/main/en/grpo_trainer" target="_blank">TRL GRPOTrainer docs</a> + <a href="https://web.stanford.edu/~jurafsky/slp3/" target="_blank">Jurafsky & Martin, SLP3 Ch. 7-10</a> + <a href="https://arxiv.org/abs/2005.14165" target="_blank">Brown et al. (2020), GPT-3: Few-Shot Learners</a> + <a href="https://arxiv.org/abs/2201.11903" target="_blank">Wei et al. (2022), Chain-of-Thought Prompting</a> + <a href="https://arxiv.org/abs/2402.03300" target="_blank">Shao et al. (2024), DeepSeekMath (GRPO)</a> + <a href="https://arxiv.org/abs/2411.15124" target="_blank">Lambert et al. (2024), Tülu 3 (RLVR)</a> + <a href="https://arxiv.org/abs/2504.13837" target="_blank">Yue et al. (2025), Does RL Really Incentivize Reasoning?</a>'
video: 
categories: LLM/Agents
lectureVersion: current
tags:
- 6LLMAgents
- Generative
- Training
- RL
---


- Notebooks to try:

<a href="https://colab.research.google.com/drive/1i7aDdW-RHNNPgCCHW9Ky858riqYbrN1R"> DPO Notebook </a> + <a href="https://colab.research.google.com/drive/1hydSFeV8hpTleZTp5EwpoDCjrqRp8JQ7"> GRPO Notebook </a>


# Study Guide: How Are Large Language Models Trained?


---


## running notebook 

<a href="https://colab.research.google.com/drive/1i7aDdW-RHNNPgCCHW9Ky858riqYbrN1R"> DPO Notebook </a> (Zephyr-7B SFT model + UltraFeedback preference pairs, QLoRA, one Colab GPU)

<a href="https://colab.research.google.com/drive/1hydSFeV8hpTleZTp5EwpoDCjrqRp8JQ7"> GRPO Notebook </a> (math problems + answer keys, rule-based checker, TRL GRPOTrainer)



## Short-Answer Quiz

**Instructions:** Answer the following questions in 2-3 sentences based on the provided source material.

1. Why does the lecture describe an LLM as "both" a discriminative and a generative model?
2. How is the pretraining loss of a GPT-style LLM related to maximum likelihood estimation, and why is pretraining called "self-supervised"?
3. What does the temperature τ do during decoding, and what happens as τ → 0 and τ → ∞?
4. How does in-context learning differ from every other learning algorithm covered earlier in the course?
5. How does the supervised fine-tuning (SFT) loss differ from the pretraining loss?
6. Name two limitations of SFT (imitation) that motivate learning from evaluation instead.
7. In the policy-gradient (REINFORCE) update, how is the step similar to an SFT step, and how is it different?
8. What is the Bradley–Terry reward model, and how is it connected to logistic regression?
9. Why does DPO not need a separate reward model or online sampling?
10. In GRPO, how is the advantage of each sampled response computed, and what happens when all G responses are correct (or all wrong)?

---

## Answer Key

1. **An LLM is a classifier over the next token, run in a loop to generate text.** At each step it outputs a softmax distribution over a vocabulary of about 100,000 tokens (a discriminative step); sampling a token, appending it, and repeating turns that classifier into a generative model of whole pieces of text.

2. **The pretraining loss is the average negative log-likelihood of each true next token, L(θ) = −(1/N) Σ log pθ(xₜ | x<ₜ), so minimizing it is exactly maximum likelihood on the corpus.** It is self-supervised because the labels are free: the "correct" next word is already in the text, so no human annotation is needed.

3. **Temperature divides the logits by τ before the softmax, pₜ(i) ∝ exp(zₜ,ᵢ/τ), making the distribution sharper (τ < 1) or flatter (τ > 1).** As τ → 0 sampling becomes greedy (always the arg-max token); as τ → ∞ the distribution becomes uniform. τ > 0 is why ChatGPT answers differently each time.

4. **In-context learning adapts the model's behavior from k input→output examples placed in the prompt, with no gradient step and no change to the weights θ.** Every earlier algorithm in the course learns by changing weights; with ICL the "learning" happens inside a single forward pass.

5. **The SFT loss is the same cross-entropy as pretraining, but computed only on the response tokens.** The prompt tokens serve as context only (loss masking), while the network, softmax head, and optimizer are unchanged; only the data is new.

6. **Any two of:** (a) costly supervision — someone must write a full, high-quality response for every prompt; (b) every token counts the same, so a typo is penalized like a factual error; (c) a ceiling — the model can only copy the demonstrator; (d) judging is easier than writing, so a score is far cheaper to obtain than a demonstration.

7. **Both steps follow the gradient of log pθ(y | x).** In SFT, y is a human-written label with weight 1; in policy gradient, y is the model's own sampled response and the step is weighted by its reward R(x, y), so good samples are pushed up and bad ones pushed down.

8. **The Bradley–Terry model gives each response a hidden score r_φ(x, y) and models the probability that humans prefer y_w over y_l as σ(r_φ(x, y_w) − r_φ(x, y_l)).** It is trained by minimizing the log loss on observed comparisons, which is logistic regression where the "score" is a difference of two learned scores and the label is "w beats l."

9. **The KL-regularized RLHF objective has a closed-form optimum, which lets the reward be rewritten as β log(πθ(y|x)/π_ref(y|x)) + const, a policy's implicit reward.** Plugging this into the Bradley–Terry loss cancels the constant, so the policy can be trained directly on fixed preference pairs (offline) — the policy is the reward model.

10. **Âᵢ = (rᵢ − mean(r₁,…,r_G)) / (std(r₁,…,r_G) + ε), and the same Âᵢ is applied to every token of response oᵢ.** If all responses are correct or all are wrong, every advantage is 0 and the question produces no gradient; learning happens only where the model sometimes succeeds.

---

## Essay Questions

**Instructions:** The following questions are designed for a more in-depth, essay-style response. Answers are not provided.

1. **Trace the full training pipeline from base model to reasoning model** (pretraining → SFT → preference alignment → RL with verifiable rewards). For each stage, answer the course's four questions: What is the data? What is the loss? How is it optimized? What does the model learn?

2. **Explain the lecture's claim that "every stage trains the same network by gradient descent on log pθ(text)."** Using the update θ ← θ + α Σ w(y) ∇θ log pθ(y | x), describe where y comes from and what the weight w(y) is for pretraining, SFT, RLHF, DPO, and GRPO.

3. **Compare supervised learning with reinforcement learning** in terms of feedback, data distribution, granularity, and objective. Then explain the two stability tricks used whenever the training data is self-generated (PPO clipping and the KL penalty), and relate the KL penalty to L2 regularization.

4. **Compare RLHF (PPO) and DPO** in terms of the reward model, sampling during training, the number of models held in memory, hyperparameter tuning, and their main weaknesses (reward hacking vs. being limited to the given pairs). When would you choose each?

5. **Discuss reasoning models such as OpenAI o1 and DeepSeek-R1.** Why is reinforcement learning with verifiable rewards preferred over SFT on worked solutions for reasoning? What behaviors emerged in DeepSeek-R1-Zero (longer thinking, re-checking, the "aha moment"), and what open questions remain (new reasoning vs. sharper sampling, diversity collapse, sparse rewards)?

---

## Glossary of Key Terms

| Term | Definition |
|------|------------|
| **Advantage (A)** | How much better a response is than expected: A = R − b, the reward minus a baseline. Subtracting a baseline leaves the expected gradient unchanged but greatly lowers its variance. |
| **Alignment** | Post-training a language model to be helpful, honest, and harmless (HHH), learned from human judgments rather than a hand-written loss. |
| **Autoregressive Generation** | Generating text one token at a time, each conditioned on the tokens before it: compute pₜ, pick xₜ, append, repeat until `<eos>`. |
| **Base Model** | A model after pretraining only; it continues a prompt the way web text would rather than trying to help the user. |
| **Bradley–Terry Model** | A model of pairwise preferences, P(y_w ≻ y_l \| x) = σ(r_φ(x, y_w) − r_φ(x, y_l)), used to train a reward model by logistic regression on comparison pairs. |
| **Chain-of-Thought (CoT)** | Prompting the model to write intermediate reasoning steps before the final answer, giving it a scratchpad and effectively more computation; can be few-shot or zero-shot ("Let's think step by step"). |
| **Direct Preference Optimization (DPO)** | A method that trains the policy directly on preference pairs using a logistic loss on implicit rewards β log(πθ/π_ref), with no reward model and no online sampling. |
| **Emergent Abilities** | Skills that stay at chance below a model-size threshold and then jump, and are not predictable from smaller models. |
| **GRPO (Group Relative Policy Optimization)** | A policy-gradient method that samples G responses per question and uses the group's normalized rewards as the advantage, replacing PPO's learned value network (critic). |
| **In-Context Learning (ICL)** | Learning a task from input→output examples placed in the prompt, with frozen weights and no gradient updates. |
| **Instruction Tuning** | Supervised fine-tuning on instruction–response pairs (and multi-turn chat data) that turns a text continuer into an assistant and enables zero-shot generalization to new instructions. |
| **KL Penalty** | A term −β KL(πθ ‖ π_ref) that keeps the policy close to a frozen reference model, preserving fluent language and limiting reward hacking. |
| **Language Model (LM)** | A model that assigns probabilities to token sequences via the chain rule, P(x₁,…,x_T) = Π P(xₜ \| x<ₜ). |
| **Policy (πθ)** | In RL, a distribution over actions given the state; for an LLM, the state is the prompt plus tokens so far, the action is the next token, and πθ is the same softmax classifier. |
| **Policy Gradient (REINFORCE)** | The gradient estimator ∇θJ = E[R(x, y) ∇θ log πθ(y \| x)], which increases a reward that can only be sampled, not differentiated. |
| **PPO Clipping** | Limiting each token's probability ratio ρₜ = πθ/π_old to [1 − ε, 1 + ε] so a single update cannot move the policy too far. |
| **Pretraining** | Training on trillions of tokens of unlabeled web text with the next-token cross-entropy loss (maximum likelihood); produces a base model. |
| **Reward Hacking** | When the policy finds outputs the learned reward model over-rates (long, confident, flattering) without being genuinely better. |
| **Reward Model** | The LLM backbone with the softmax head replaced by a linear layer that outputs one number; a cheap, differentiable proxy for human preference. |
| **RLHF** | Reinforcement Learning from Human Feedback: collect comparisons, train a reward model, then maximize that reward with PPO under a KL leash to the SFT model. |
| **RLVR** | Reinforcement Learning with Verifiable Rewards: the reward comes from a program that checks the answer (answer key, unit tests), so no human label or learned reward model is needed. |
| **Scaling Laws** | The empirical finding (Kaplan et al., 2020) that test loss falls as a power law in compute, data, and parameters. |
| **Supervised Fine-Tuning (SFT)** | Continuing training of the pretrained network on (prompt, response) pairs with cross-entropy on the response tokens only; equivalent to behavior cloning. |
| **Temperature (τ)** | A decoding parameter that divides the logits before the softmax; low τ is sharper (greedy as τ → 0), high τ is flatter (uniform as τ → ∞). |
| **Tokenization** | Mapping text to integer IDs from a vocabulary of common character chunks (about 50k–128k subword tokens); the model sees tokens, not letters. |
