---
LOrder: 400
layout: post
title: deep RL Gym
lecture: S3-L19z-RL_Gyms
lectureVersion: current
notes: 'Mnih et al. (2015), <a href="https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf" target="_blank">Human-level Control through Deep Reinforcement Learning</a> · Lambert et al. (2022), <a href="https://huggingface.co/blog/rlhf" target="_blank">Illustrating Reinforcement Learning from Human Feedback (RLHF)</a> · Shao et al. (2024), <a href="https://arxiv.org/abs/2402.03300" target="_blank">DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models</a>'
morenotes: TA Speaker
video:    
categories: library
platform: true
tags:
- RL
- Platform
- 6LLMAgents
- Training
---

### In this lecture, we cover:
- Why RL Gyms: RL learns by trial and error from trajectories of states, actions, and rewards (AlphaGo Zero needed 4.9M self-play games), which is inefficient and unsafe in the real world
- Physical RL Gyms
  + Atari (discrete control): 472 games; DQN reached human-level or better performance in 29 of 49 benchmarked games
  + MuJoCo (continuous control): led to TRPO and PPO
  + Isaac Lab (embodied, sim-to-real gap)
  + CleanRL single-file implementations: DQN on Breakout, PPO on HalfCheetah, PPO on the ANYmal quadruped in Isaac Gym
- RL for post-training LLMs
  + Pipeline: pre-training, SFT, then RLHF (helpfulness, harmlessness) or RLVR (math and coding reasoning); language as the missing action space
  + RLHF in three stages (SFT, reward modeling, RL post-training); OpenRLHF and its GPU-memory tips (remote reward model, or DPO)
  + Hands-on DPO: TRL pipeline on Zephyr-7B-sft, offline vs. online RL, UltraFeedback preference pairs, DPO training curves
  + RLVR with SimpleGRPO: format and correctness rewards on GSM8k (Qwen-1.5B)
  + RL fine-tuning frameworks: TRL, SimpleGRPO, Llama-Factory, Verl, Tinker
- A natural next question: great RL Gyms inspire great RL algorithms, so what is missing in today's gyms?

# Study Guide for Reinforcement Learning Gyms

## Quiz: Short-Answer Questions

Answer each of the following questions in 2-3 sentences, based on the provided source material.

1. Why are simulated environments, or "RL Gyms," essential for the development of Reinforcement Learning algorithms?
2. What was DeepMind's primary goal in using the Atari environment, and what key algorithm was developed to solve it?
3. Explain the fundamental difference between the action spaces of the Atari and MuJoCo environments.
4. What specific challenge in robotics prompted the development of the Isaac Lab environment?
5. What is the stated purpose of CleanRL, and what is a notable feature of its implementation?
6. Describe the three main stages involved in the post-training of Large Language Models (LLMs) as outlined in the document.
7. From the perspective of Reinforcement Learning, what "missing piece" do natural language tasks provide that was absent in earlier gyms like Atari and MuJoCo?
8. What are the three stages of Reinforcement Learning from Human Feedback (RLHF)?
9. How does Reinforcement Learning from Verifiable Reward (RLVR) differ from RLHF in its approach to rewards?
10. What is the SimpleGRPO framework, and what two types of rewards does it use for training?
11. In the hands-on DPO practice, which model and library are used, and what is the starting model?
12. How does DPO differ from online RL such as PPO?
13. How are the chosen and rejected responses built from the UltraFeedback dataset?
14. When OpenRLHF needs too much GPU memory, what two options does the lecture suggest?

## Answer Key

1. RL Gyms are essential because Reinforcement Learning requires collecting massive amounts of trajectory data through trial and error, citing AlphaGo Zero's 4.9 million self-play games as an example. Performing this scale of learning in the real world is both inefficient and unsafe, making simulated environments a necessary alternative.
2. DeepMind aimed to develop a general-purpose learning algorithm for AGI that could solve diverse tasks at a human level. The Atari testbed, with its 472 different games, was used for this purpose, leading to the creation of the Deep Q-Network (DQN) algorithm.
3. The Atari environment has a discrete action space, such as moving "left" or "right" in the game Pong. In contrast, the MuJoCo environment was created for continuous control problems, where actions can be arbitrary degrees of movement or rotation, more closely mirroring real-world control.
4. The Isaac Lab environment was designed to address the notorious gap between simulation and reality in robotics. While environments like MuJoCo used simplified robots, Isaac Lab focuses on simulating real-world scenarios to better bridge this gap.
5. CleanRL is designed to provide high-performance, single-file implementations of popular RL algorithms to help researchers understand their code implementation. A notable feature is its conciseness; for example, its ppo_atari.py file contains only 340 lines of code.
6. The three stages are: Stage 1, pre-training with self-supervised learning on next-token prediction; Stage 2, supervised fine-tuning (SFT) on human-annotated responses; and Stage 3, either RL from Human Feedback (RLHF) to improve helpfulness or RL from Verifiable Reward (RLVR) to incentivize reasoning in tasks like math and coding.
7. Natural language tasks fill the missing piece of an essential action space for humans: language. Previous gyms like Atari, Go, and DOTA covered action spaces defined by human-designed games, while MuJoCo and Isaac Lab covered continuous control for robotics, leaving language as an unexplored action space.
8. The three stages of RLHF are: Stage 1, Supervised Fine-Tuning (SFT); Stage 2, Reward Modeling, where a model learns from human rankings of responses; and Stage 3, RL post-training, where the LLM is fine-tuned using the learned reward model.
9. RLVR uses rule-based, objective rewards, similar to those in Atari or Go, instead of the subjective and potentially noisy human preferences used in RLHF. RLVR verifies the correctness of a model's answer against a ground-truth answer to solve hard problems like math and coding.
10. SimpleGRPO is presented as a training framework for understanding RLVR. It uses a Format Reward to enforce a chain-of-thought reasoning structure and a Correctness Reward to verify the model's final answer against the ground-truth answer.
11. The practice uses the TRL pipeline to fine-tune Zephyr-7B-sft with DPO, in a Google Colab notebook that needs a GPU (e.g., an A100 via Colab Pro). Zephyr-7B-sft is a Mistral-7B derivative already fine-tuned on the UltraChat dataset, so it serves as the SFT base.
12. DPO is offline: it learns from a static preference dataset without generating new samples during training. Online RL such as PPO generates new samples from the current policy as it trains.
13. UltraFeedback has 60k+ prompts with GPT-4-labeled responses. For each prompt, the chosen response (y_w) is the one with the highest overall score, and the rejected response (y_l) is a randomly selected remaining response.
14. Use a remote reward model to reduce GPU usage, or use DPO for a further reduction, with some performance trade-offs.

## Essay Questions

Construct a detailed, essay-format response for each of the following prompts, synthesizing information from the source material.

1. Trace the evolution of RL Gyms from Atari to Isaac Lab. For each environment (Atari, MuJoCo, Isaac Lab), describe the specific type of control problem it was designed to address, the algorithms associated with it (DQN for Atari; TRPO and PPO for MuJoCo; and, for Isaac Lab, the CleanRL example of PPO training the ANYmal quadruped), and how it paved the way for the subsequent environment.
2. Explain the complete pipeline for post-training a Large Language Model using Reinforcement Learning. Detail the purpose and methodology of each stage, from pre-training and supervised fine-tuning through the final reinforcement learning stage, and differentiate between the goals of RLHF and RLVR.
3. Compare and contrast Reinforcement Learning from Human Feedback (RLHF) and Reinforcement Learning from Verifiable Reward (RLVR). Discuss the motivations for each, the methods used to generate rewards, the types of tasks each is best suited for, and the limitations of RLHF that RLVR aims to overcome.
4. The document states, "Great RL Gyms inspire great algorithms." Using Atari/DQN and MuJoCo/PPO as primary examples, elaborate on this argument. How did the specific challenges and characteristics of each gym lead to the development of its corresponding breakthrough algorithm?
5. Discuss the role of language as an "essential action space for humans" within the context of Reinforcement Learning environments. Explain why this was considered a "missing piece" and how frameworks like OpenRLHF and SimpleGRPO facilitate the exploration of this action space for LLMs.

## Glossary of Key Terms

| Term | Definition |
|------|------------|
| Atari | A testbed of 472 diverse video games used to develop and evaluate general-purpose learning algorithms. It features a discrete control action space and was instrumental in the development of the Deep Q-Network (DQN). |
| CleanRL | A project providing single-file, high-performance implementations of popular RL algorithms, designed to make understanding RL code implementation easier. An example is ppo_atari.py containing only 340 lines of code. |
| Correctness Reward | A type of reward used in RLVR, where the correctness of a model-generated answer is verified against a ground-truth answer. |
| DPO (Direct Preference Optimization) | An offline alternative used in the hands-on practice: it learns from a static dataset of chosen/rejected response pairs without generating new samples, and needs less GPU memory than full RLHF, with some performance trade-offs. Training curves track reward accuracies, margins, and chosen/rejected rewards. |
| Deep Q-Network (DQN) | An end-to-end RL algorithm developed by DeepMind to solve Atari games. It takes raw video frames as input and outputs discrete control actions, achieving human-level or better performance in 29 of 49 benchmarked games (Mnih et al. 2015). |
| Format Reward | A type of reward used in RLVR that enforces a specific output structure, such as requiring an LLM to produce a chain-of-thought reasoning process. |
| Isaac Lab | An RL Gym designed to simulate real-world scenarios for robotics, created to address the "notorious gap between simulation and reality." |
| Llama-Factory | An RL fine-tuning framework that, with ktransformers, can fine-tune trillion-parameter models. |
| MuJoCo | An RL testbed proposed for continuous control problems, which contrasts with Atari's discrete action space. Its challenges led to the development of algorithms like TRPO and PPO. |
| Offline vs. Online RL | Online RL (e.g., PPO) generates new samples during training; offline methods (e.g., DPO) learn only from a fixed, pre-collected dataset. |
| OpenRLHF | A stable and scalable RLHF implementation that includes a built-in Reward Modeling pipeline. It is designed for modeling human preference but can be adapted for other tasks. It typically needs substantial GPU memory; a remote reward model or DPO reduces this. |
| PPO (Proximal Policy Optimization) | An RL algorithm that emerged from efforts to solve MuJoCo's continuous control problems. It has since become the de facto RL algorithm in many settings, including the post-training of LLMs. |
| Reinforcement Learning (RL) | A machine learning paradigm that involves learning through trial and error by collecting trajectories of states, actions, and rewards from an environment. |
| Reinforcement Learning from Human Feedback (RLHF) | A three-stage process to post-train LLMs to improve helpfulness and reduce harmfulness. It involves SFT, learning a reward model from human preferences, and then fine-tuning the LLM with that reward model. |
| Reinforcement Learning from Verifiable Reward (RLVR) | A method of reinforcement fine-tuning that uses rule-based, verifiable rewards (like correctness) to solve hard problems such as math and coding, where human preference can be subjective or noisy. |
| Reward Modeling | The second stage of RLHF. Several model outputs for a prompt are ranked by humans, and a reward model learns these grading criteria to score responses in the final RL stage. |
| RL Gyms | Simulated environments of the real world where RL agents can learn through massive amounts of trial and error without the inefficiency and safety risks of real-world interaction. |
| SimpleGRPO | An RLVR training framework noted for its clarity and ease of understanding. It is used as an example for training a model on the GSM8k math dataset. |
| Supervised Fine-Tuning (SFT) | The second stage of the overall LLM training pipeline (and the first stage of the RLHF process). It involves fine-tuning a pre-trained model on human-annotated responses. |
| Tinker | A recent framework from Thinking Machines focused on stable and robust PPO fine-tuning for LLMs. |
| TRL | An RL fine-tuning framework from Hugging Face. |
| UltraFeedback | A preference dataset of 60k+ prompts with GPT-4-labeled responses; in the DPO practice, the highest-scoring response is "chosen" and a random remaining one is "rejected." |
| Verl | An RL fine-tuning framework that provides support for Multi-Turn Tool Calling. |
| Zephyr-7B-sft | A Mistral-7B derivative already fine-tuned on UltraChat, used as the SFT base model for DPO training with TRL. |
