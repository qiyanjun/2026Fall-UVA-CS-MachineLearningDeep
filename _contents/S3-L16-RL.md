---
LOrder: 320
layout: post
title: Reinforcement Learning and for Post Training LLMs
lecture: S3-L19b-RL-RLHF-GRPO
extra: true
lectureVersion: current
notebooks: '<a href="https://colab.research.google.com/github/qiyanjun/2026Fall-UVA-CS-MachineLearningDeep/blob/main/notebook/L19b_DPO.ipynb" target="_blank">notebook/L19b_DPO.ipynb</a> · <a href="https://colab.research.google.com/github/qiyanjun/2026Fall-UVA-CS-MachineLearningDeep/blob/main/notebook/L19b_Reasoning_Models_using_GRPO.ipynb" target="_blank">notebook/L19b_Reasoning_Models_using_GRPO.ipynb</a> · <a href="https://colab.research.google.com/github/qiyanjun/2026Fall-UVA-CS-MachineLearningDeep/blob/main/notebook/L19-RL-RLHF-GRPO-notebook-policy-gradient-to-grpo.ipynb" target="_blank">notebook/L19-RL-RLHF-GRPO-notebook-policy-gradient-to-grpo.ipynb</a>'
video: 
notes: 'Sutton & Barto (2018), <a href="http://incompleteideas.net/book/the-book-2nd.html" target="_blank">Reinforcement Learning: An Introduction</a> · Weng (2018), <a href="https://lilianweng.github.io/posts/2018-02-19-rl-overview/" target="_blank">A (Long) Peek into Reinforcement Learning</a> · Mnih et al. (2015), <a href="https://www.nature.com/articles/nature14236" target="_blank">Human-level control through deep reinforcement learning</a> · Schulman et al. (2017), <a href="https://arxiv.org/abs/1707.06347" target="_blank">Proximal Policy Optimization Algorithms</a> · Ouyang et al. (2022), <a href="https://arxiv.org/abs/2203.02155" target="_blank">Training language models to follow instructions with human feedback</a> · Rafailov et al. (2023), <a href="https://arxiv.org/abs/2305.18290" target="_blank">Direct Preference Optimization: Your Language Model is Secretly a Reward Model</a> · Shao et al. (2024), <a href="https://arxiv.org/abs/2402.03300" target="_blank">DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models</a> · DeepSeek-AI (2025), <a href="https://arxiv.org/abs/2501.12948" target="_blank">DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning</a>'
categories: structured
tags:
- RL
---

### In this lecture, we cover:
- Why RL for LLMs: cross-entropy rewards copying the data token by token, but what we want (helpful, correct, safe) is a property of the whole answer, scored only after it is written; learning from a score that comes after a sequence of choices is reinforcement learning
- What is reinforcement learning?
  + Supervised (map x → y), unsupervised (find structure in x), and reinforcement learning (maximize future rewards over many steps by interacting with an environment)
  + Examples: Atari 2600 (raw pixels, valid moves, game score; same model for ~50 games) and AlphaGo / AlphaZero (board state, valid moves, win or lose)
  + What is special about RL: trial-and-error learning, the agent chooses its own training data, delayed effects of actions, and credit assignment (which of 100 moves, or 500 tokens, deserves the credit?)
  + Elements of RL: a policy π(a\|s), a reward r (what is good now), a value function V(s) or Q(s, a) (what is good in the long run), and an optional model P(s′\|s, a) (model-based vs. model-free RL)
  + The Markov decision process (MDP): states, actions, rewards, transitions, the Markov property, and the discounted return with 0 ≤ γ &lt; 1
  + A grid-world example (treasure chest worth 100, γ = 0.5): evaluating a policy, and exploration vs. exploitation (slot machines)
- Value-based RL: V, Q, Bellman, Q-learning
  + The classic algorithms: dynamic programming (policy iteration, value iteration; needs a full model), Monte Carlo (complete episodes), temporal-difference (TD) learning such as Q-learning (every step, no model), and policy gradients
  + Value functions V(s) and Q(s, a), the Bellman equation, and the Bellman optimality equation for V\* and Q\*
  + Monte Carlo (unbiased but high variance, episodic tasks only) vs. TD learning (update after every step from the successor's value)
  + ε-greedy exploration (LLM analogue: sampling with temperature &gt; 0 instead of greedy decoding)
  + Q-learning step by step on the grid world (rounds 1 and 2, then convergence to Q values 100, 50, 25, 12.5, 6.25) and the greedy policy π(s) = argmax<sub>a</sub> Q(s, a)
  + Deep Q-learning (DQN): a CNN approximates Q(s, a; w) from pixels, trained by regression toward the TD target, with experience replay and a frozen target network w<sup>−</sup>
- Policy gradients: REINFORCE, actor-critic, PPO
  + Policy-based vs. value-based RL: learning π<sub>θ</sub>(a\|s) directly handles stochastic policies and huge action spaces (a 50k-token vocabulary); an LLM is already a stochastic policy over tokens
  + Policy gradient in one line via the log-derivative trick ∇p = p ∇log p: sample, score, then raise the log-probability of what you did in proportion to the reward; with data tokens and R = 1 it is the next-token cross-entropy gradient
  + REINFORCE (Monte Carlo policy gradient), subtracting a baseline b(s<sub>t</sub>) to lower variance, and the advantage
  + Actor-critic: a critic V(s; w) or Q learned alongside the policy (actor), with bootstrapped updates at every step
  + One table: every method multiplies ∇log π by a weight W<sub>t</sub> (return, return − baseline, advantage, group-normalized reward, or 1 for SFT)
  + PPO: the probability ratio ρ<sub>t</sub>(θ) and the clipped objective (ε ≈ 0.2), the RL step of InstructGPT / ChatGPT
  + Quiz: why an action-independent baseline does not bias the gradient
- RLHF and DPO: aligning an LLM
  + An LLM is a policy: state = prompt + tokens so far, action = next token, policy = the softmax at the last position, episode = one full answer, reward r(x, y) once at the end
  + Post-training in three steps: supervised instruction fine-tuning on (X, Y), RLHF with PPO or DPO on preference data (X, Y<sub>w</sub>, Y<sub>l</sub>), then RL with verifiable rewards (RLVR)
  + The three phases of RLHF: an instruction-tuned base model, preference collection and reward model training (y<sub>w</sub> should score higher than y<sub>l</sub>), and policy optimization
  + The KL penalty to the reference model π<sub>ref</sub>, which keeps the policy from producing gibberish that fools the reward model
  + InstructGPT (scaling RLHF to 30k tasks) and RLHF in practice (reward-model overfitting, memory and compute cost, numerical instabilities)
  + Direct Preference Optimization (DPO): the RLHF objective has a closed-form solution, so the reward can be written in terms of the policy, giving a classification loss on preference pairs
- RL with verifiable rewards: GRPO, DeepSeek-R1
  + RLVR (Tülu 3): a verification function on math and coding problems gives reward 1 if correct and 0 otherwise
  + DeepSeek-R1 and R1-Zero: pure RL from DeepSeek-V3-Base with format rewards (&lt;think&gt; / &lt;answer&gt; tags) and rule-based accuracy rewards (does it compile, does it pass unit tests)
  + GRPO vs. PPO: no value model; the advantage of each answer is computed relative to a group of answers to the same prompt
  + GRPO in our notation: group-normalized advantage, PPO-clip, and a KL term to π<sub>ref</sub>; worked example with G = 4 and rewards 1, 0, 0, 1
  + The "aha moment" of DeepSeek-R1-Zero, and distilling reasoning from DeepSeek-R1 (teacher) into a smaller student (Qwen-32B)
- Takeaways and references

*(SFT, LoRA, and an introduction to RLHF and DPO are in "Transformer Post-training"; hands-on RL environments and tooling are in "deep RL Gym".)*


# Study Guide: Reinforcement Learning and for Post Training LLMs

---

## Quiz: Short-Answer Questions

**Instructions:** Provide a concise answer (2-3 sentences) for each of the following questions.

1. Why is next-token cross-entropy not enough to make an LLM helpful, correct, and safe, and why does that call for reinforcement learning?
2. Name the four elements of RL, and explain the difference between a reward and a value.
3. What is credit assignment, and why is it a problem both for a game player and for an LLM writing a 500-token answer?
4. Compare Monte Carlo and temporal-difference (TD) learning. Which one needs the episode to end?
5. On the grid world (γ = 0.5, deterministic moves), after rounds 1 and 2 the only non-zero entries are Q̂(&lt;2,1&gt;, north) = 100 and Q̂(&lt;2,2&gt;, north) = 50. Round 3 starts at &lt;1,1&gt; and moves east to &lt;2,1&gt; with reward 0. What is the new Q̂(&lt;1,1&gt;, east)? What would Q̂(&lt;1,1&gt;, south) be if it had moved south to &lt;1,2&gt; instead?
6. In the same example, why did &lt;2,2&gt; get a non-zero value only in round 2, not in round 1?
7. What two tricks make deep Q-learning (DQN) stable, and what problem does each one fix?
8. Why does LLM training use policy gradients rather than value-based methods such as Q-learning?
9. For a fixed state s, why does subtracting a baseline b(s) that does not depend on the action leave the expected policy gradient unchanged? A group of 4 answers gets rewards 1, 1, 1, 0; with the mean as the baseline, what weight does each answer get?
10. In PPO, what do the ratio ρ<sub>t</sub>(θ) and the clip do when Â &gt; 0 and when Â &lt; 0?
11. In RLHF, why is a reward model trained, and why is a KL penalty to π<sub>ref</sub> added?
12. How does DPO remove the reward model and the RL loop?
13. GRPO samples G = 4 answers to one prompt with rewards 1, 0, 0, 1. Compute each answer's advantage, and say what GRPO no longer needs compared with PPO.

---

## Answer Key

1. **Cross-entropy rewards copying the data token by token, but helpfulness, correctness, and safety are properties of the whole answer.** There is no single "right next token" for them, only a score after the answer is written, and learning from a score that comes after a sequence of choices is reinforcement learning.

2. **The elements are a policy π(a\|s), a reward r, a value function V(s) or Q(s, a), and (optionally) a model of the environment P(s′\|s, a).** The reward says what is good now; the value is the total expected reward from a state (or state-action pair), so it says what is good in the long run.

3. **Credit assignment is deciding which of many earlier actions deserves the credit for a reward that arrives later.** Actions affect the immediate reward and all later rewards, so when a game is won after 100 moves, or an answer is judged after 500 tokens, the single final score must be spread back over all the choices.

4. **Monte Carlo waits until an episode ends and averages the observed returns, so it only works for episodic tasks; it is unbiased but has high variance.** TD learning, like DP, learns from the values of successor states and updates after every step, so it works for continuing tasks and is usually faster than MC; like MC, it needs no model.

5. **Q̂(&lt;1,1&gt;, east) = 0 + 0.5 × max<sub>a′</sub> Q̂(&lt;2,1&gt;, a′) = 0.5 × 100 = 50.** Moving south to &lt;1,2&gt; gives Q̂(&lt;1,1&gt;, south) = 0 + 0.5 × 0 = 0, because every entry for &lt;1,2&gt; is still 0.

6. **The value of the chest reaches &lt;2,2&gt; only after &lt;2,1&gt; already has a non-zero Q value.** Round 1 started at &lt;1,1&gt; and never visited &lt;2,2&gt;; it set Q̂(&lt;2,1&gt;, north) = 100 at its last step. In round 2 the agent moved north from &lt;2,2&gt; to &lt;2,1&gt;, so the update read that 100 and set Q̂(&lt;2,2&gt;, north) = 0 + 0.5 × 100 = 50. Values spread backward one step per visit.

7. **Experience replay and a target network.** Experience replay stores transitions (s, a, r, s′) in a buffer and trains on random mini-batches, breaking the correlation between consecutive frames; the target network w<sup>−</sup> is a frozen copy of w refreshed every C steps, so the regression target does not move every step.

8. **An LLM is already a stochastic policy over tokens, with a huge action space of ~50k tokens.** Value-based RL acts by an argmax over Q(s, a) and has no explicit policy, which is hard with many actions; policy-based RL learns π<sub>θ</sub>(a\|s) directly and can learn stochastic policies, at the cost of local optima and high-variance gradient estimates.

9. **Writing the expectation as a sum over actions, Σ<sub>a</sub> π(a\|s) ∇log π(a\|s) b(s) = b(s) ∇ Σ<sub>a</sub> π(a\|s) = b(s) ∇1 = 0.** So the baseline lowers variance without changing the expected gradient (if b depended on a, it could no longer be pulled out of the sum, and this argument fails). With rewards 1, 1, 1, 0 the mean is 0.75, so the weights are +0.25, +0.25, +0.25, and −0.75.

10. **ρ<sub>t</sub>(θ) = π<sub>θ</sub>(a<sub>t</sub>\|s<sub>t</sub>) / π<sub>θold</sub>(a<sub>t</sub>\|s<sub>t</sub>) measures how far the new policy has moved, and the clip limits it to [1 − ε, 1 + ε] (ε ≈ 0.2).** When Â &gt; 0 PPO raises the action's probability, but by at most a factor 1 + ε; when Â &lt; 0 it lowers it, but by at most 1 − ε. This lets PPO reuse each batch for several steps without a too-large step wrecking the policy.

11. **Asking humans for every score is expensive, so a separate reward model r<sub>φ</sub>(x, y) is trained on human comparisons to score the winning answer y<sub>w</sub> above the losing answer y<sub>l</sub>.** The policy then maximizes this reward minus a KL penalty to the reference model π<sub>ref</sub>, which stops RL fine-tuning from drifting to gibberish that fools the reward model.

12. **The RLHF objective (maximize reward minus a KL penalty to π<sub>ref</sub>) has a closed-form solution, so the reward can be written in terms of the policy.** Because the reward-model loss only needs the difference of rewards, r(x, y<sub>w</sub>) − r(x, y<sub>l</sub>) = β[log(π<sub>θ</sub>(y<sub>w</sub>\|x)/π<sub>ref</sub>(y<sub>w</sub>\|x)) − log(π<sub>θ</sub>(y<sub>l</sub>\|x)/π<sub>ref</sub>(y<sub>l</sub>\|x))], and plugging this into −log σ(·) gives a classification loss that connects preference data to the LM parameters directly, with no reward model and no RL loop.

13. **The mean reward is 0.5 and the standard deviation is 0.5, so Â = (r − 0.5) / 0.5 = +1, −1, −1, +1.** All tokens of an answer share its advantage. Because the group mean serves as the baseline, GRPO needs no value model (critic), which PPO must train and run; this makes it much more efficient for scaling up LLM training.

---

## Essay Questions

**Instructions:** Prepare a detailed, essay-format response for each of the following prompts. (Answers not provided).

1. **Trace the grid-world example from policy evaluation to Q-learning.** Define the MDP (states, actions, rewards, γ = 0.5), evaluate the given policy, run Q-learning by hand until the values converge (100, 50, 25, 12.5, 6.25), and explain how ε-greedy exploration and, for large state spaces, deep Q-learning extend the method.

2. **Explain the single policy-gradient template behind REINFORCE, actor-critic, PPO, GRPO, and supervised fine-tuning.** Derive the gradient with the log-derivative trick, and compare the weight W<sub>t</sub> that each method multiplies ∇log π by, the extra model it needs, and how it controls variance and step size.

3. **Map reinforcement learning onto an LLM and walk through the three phases of RLHF.** Define state, action, policy, transition, episode, and reward for a language model; then describe instruction tuning, reward model training on preference pairs, and PPO with a KL penalty, and discuss the practical difficulties of RLHF.

4. **Compare RLHF with PPO, DPO, and GRPO with verifiable rewards.** For each, describe the data it needs, the models it trains or keeps in memory, and its loss; then explain how DeepSeek-R1-Zero used format and accuracy rewards with GRPO to elicit reasoning, and how that reasoning was distilled into a smaller model.

---

## Glossary of Key Terms

| Term | Definition |
|------|------------|
| **Reinforcement Learning (RL)** | Learning to map states to actions, by trial and error, so as to maximize a numerical reward over many steps. |
| **Policy π(a\|s)** | A map from states to actions (or a distribution over actions); may be stochastic. |
| **Reward r** | A real number for a state or state-action pair: what is good now. |
| **Value Function V(s) / Q(s, a)** | The total expected (discounted) reward from a state, or from a state-action pair, when following a policy: what is good in the long run. |
| **Markov Decision Process (MDP)** | States, actions, rewards, and transitions P(s′\|s, a) in which the future depends only on the current state. |
| **Discount Factor γ** | A number 0 ≤ γ &lt; 1 that weights a reward i steps ahead by γ<sup>i</sup> in the return. |
| **Credit Assignment** | Deciding which earlier actions (moves or tokens) deserve credit for a delayed reward. |
| **Bellman Equation** | A recursive equation relating a state's value to the reward and the values of its successor states; the optimality version defines V\* and Q\*. |
| **Policy / Value Iteration** | Dynamic programming methods that need a full model of the environment; value iteration uses the Bellman optimality equation as an update. |
| **Temporal-Difference (TD) Learning** | Learning from experience after every step by updating toward the reward plus the discounted value of the successor state. |
| **ε-greedy** | Take the greedy action with probability 1 − ε and a random action with probability ε. |
| **Q-learning** | A model-free TD method that learns Q(s, a) toward r + γ max<sub>a′</sub> Q(s′, a′) and acts by argmax. |
| **Deep Q-learning (DQN)** | Q-learning with a neural network (a CNN on pixels) for Q(s, a; w), trained with experience replay and a target network. |
| **Policy Gradient** | Directly raising the log-probability of sampled actions in proportion to their reward, using ∇p = p ∇log p. |
| **REINFORCE** | Monte Carlo policy gradient that weights ∇log π by the observed return; unbiased but high variance. |
| **Baseline / Advantage** | A value subtracted from the return to lower variance; the advantage Â ≈ Q(s, a) − V(s) says how much better than average an action was. |
| **Actor-Critic** | A policy (actor) updated in the direction suggested by a learned value function (critic), with bootstrapped updates at each step. |
| **PPO** | Proximal Policy Optimization: a policy-gradient method that clips the probability ratio to [1 − ε, 1 + ε] so the policy moves only a little per update. |
| **RLHF** | Reinforcement learning from human feedback: SFT, then a reward model from preference pairs, then PPO with a KL penalty to π<sub>ref</sub>. |
| **Reward Model** | A model r<sub>φ</sub>(x, y) trained on human comparisons to give the preferred answer a higher scalar reward. |
| **KL Penalty** | A term that keeps the fine-tuned policy close to the reference model π<sub>ref</sub>, so it does not drift into outputs that fool the reward model. |
| **DPO** | Direct Preference Optimization: rewrites the RLHF objective as a classification loss on preference pairs (y<sub>w</sub>, y<sub>l</sub>), with no reward model and no RL loop. |
| **RLVR** | RL with verifiable rewards: a checker (e.g., a math answer or unit tests) gives reward 1 if correct and 0 otherwise. |
| **GRPO** | Group Relative Policy Optimization: PPO without a critic, using the reward normalized over a group of G answers to the same prompt as the advantage. |
| **DeepSeek-R1-Zero** | A reasoning model trained by pure RL from DeepSeek-V3-Base with format and accuracy rewards. |
| **Distillation** | Training a smaller student model to match a larger teacher's output distribution, as in distilling DeepSeek-R1's reasoning into Qwen-32B. |
