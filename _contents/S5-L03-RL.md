---
layout: post
title: Reinforcement Learning
lecture: S5_RL22 
lectureVersion: current
video: 
notes: 
categories: structured
tags:
- notSupervised
- RL
- 5Theory
---



# Reinforcement Learning Study Guide

## Quiz

Answer the following questions in 2-3 sentences each, based on the provided lecture materials.

1. How does Reinforcement Learning (RL) fundamentally differ from Supervised and Unsupervised Learning?
2. Describe the four core elements of the Reinforcement Learning setup.
3. What is a Markov Decision Process (MDP) and what is its key property?
4. Explain the difference between a "reward function" and a "value function" in RL.
5. What was the significance of the TD-Gammon program developed by Gerry Tesauro?
6. Compare the primary requirements of Dynamic Programming (DP) versus Monte Carlo (MC) methods for solving RL problems.
7. What is the "explore vs. exploit" dilemma, and what is the ε-greedy policy strategy for managing it?
8. How does Temporal-Difference (TD) learning combine ideas from both Dynamic Programming and Monte Carlo methods?
9. Define Q-learning and explain why it is considered an "off-policy" algorithm.
10. What is reward shaping and what is the guiding principle for designing good reward functions?

---

## Answer Key

1. Unlike Supervised Learning which learns a map from data x to a label y, and Unsupervised Learning which learns underlying structure from unlabeled data x, Reinforcement Learning's goal is to maximize future rewards over many steps. The data for RL consists of state-action pairs and the resulting rewards, as the agent learns by interacting with an environment.

2. The core elements are the agent, which is the learner or decision-maker; the environment, which the agent interacts with; the action, which is what the agent chooses to do; and the reward and new state, which are the feedback the environment provides in response to an action.

3. A Markov Decision Process (MDP) is a discrete-time stochastic control process that formally defines the environment in an RL problem. Its key property is that every outcome (the new state and reward) depends only on the current state and the chosen action, not on any previous states or actions.

4. A reward function maps a state or state-action pair to an immediate, single numerical value called a reward. In contrast, a value function represents the total expected future reward an agent can accumulate over the long run, starting from a specific state or state-action pair.

5. TD-Gammon was a Backgammon-playing program that became a major success story for RL. It used temporal-difference learning and a neural network, training itself to a world-class level simply by playing 1.5 million games against itself and learning from the outcomes.

6. Dynamic Programming requires a perfect and complete model of the environment, including knowledge of state transition probabilities and rewards. Monte Carlo methods do not need a model of the environment; instead, they learn directly from experience (or simulated experience) by averaging sample returns from completed episodes.

7. The "explore vs. exploit" dilemma is the challenge of choosing between actions that are known to yield good rewards (exploit) and trying new, unknown actions that might lead to better rewards (explore). The ε-greedy policy is a strategy where the agent chooses the known best action with probability 1-ε, but chooses a random action with a small probability ε to ensure continued exploration.

8. TD learning combines ideas from both methods by learning directly from experience without a model, similar to MC methods. However, it updates its value estimates based on the values of successor states, a technique known as bootstrapping that is characteristic of DP.

9. Q-learning is a temporal-difference method that directly approximates the optimal state-action value function, Q*. It is considered "off-policy" because it can learn the optimal policy regardless of the policy the agent is actually following to explore the environment, as long as every state-action pair continues to be updated.

10. Reward shaping is the practice of providing rewards for achieving subgoals to guide the agent, which is useful when the primary positive reward is very "far away" or difficult to achieve. The principle for good rewards is that they should indicate what the agent needs to accomplish, not how it should be accomplished.

---

## Essay Questions

Construct detailed, essay-format answers for the following prompts, synthesizing information from across the lecture materials.

1. Discuss the historical development of Reinforcement Learning, citing the key contributions of Thorndike, Bellman, Samuel, and Watkins. Explain how each contribution added a foundational piece to the modern understanding of RL.

2. Compare and contrast the three main approaches for solving RL problems: Dynamic Programming, Monte Carlo methods, and Temporal-Difference learning. For each, describe its core idea, requirements (e.g., model of the environment), and its primary advantages and disadvantages.

3. Using the "Robot in a room" example, explain the roles of states, actions, rewards, the policy, and the value function. Describe how a discount factor (γ) affects the calculation of the value function and why it is a critical component in defining the agent's goal.

4. AlphaGo is presented as a major success for Deep Reinforcement Learning. Describe its learning pipeline, explaining how it combines Supervised Learning (SL) and Reinforcement Learning (RL) and the specific roles of the SL policy network, the value network, and the Monte Carlo Tree Search (MCTS).

5. What are the key challenges in Reinforcement Learning? Discuss at least three challenges mentioned in the lecture, such as delayed rewards, the explore vs. exploit trade-off, state representation, and reward design, providing examples for each.

---

## Glossary of Key Terms

| Term | Definition |
|------|------------|
| **Action** | A choice made by the agent from a set of available options in a given state. |
| **Agent** | The learner or decision-maker that interacts with the environment. |
| **Bellman Equation** | A fundamental equation in RL that expresses the value of a state or state-action pair in terms of the expected reward plus the discounted value of successor states. |
| **Deep Reinforcement Learning** | The combination of Reinforcement Learning with deep neural networks, where a neural network can learn the policy or value function. |
| **Discount Factor (γ)** | A value between 0 and 1 that determines the present value of future rewards. It models the trade-off between immediate and future rewards. |
| **Dynamic Programming (DP)** | A method for solving RL problems that uses value functions to structure the search for good policies, but requires a perfect model of the environment. |
| **Environment** | The external system with which the agent interacts, providing states and rewards in response to the agent's actions. |
| **ε-greedy policy** | An exploration strategy where the agent chooses the action with the highest estimated value with probability 1-ε and a random action with probability ε. |
| **Episodic Task** | A task that has a clear starting and ending point, breaking the agent's experience into distinct episodes (e.g., a game of chess). |
| **Explore vs. Exploit** | The fundamental trade-off in RL between taking actions with known high rewards (exploitation) and trying new actions to discover potentially better rewards (exploration). |
| **Markov Decision Process (MDP)** | A mathematical framework for modeling decision-making in an environment where outcomes are partly random and partly under the control of a decision-maker. It assumes that future states depend only on the current state and action. |
| **Monte Carlo (MC) Methods** | A class of RL algorithms that learn from complete episodes of experience without needing a model of the environment. They estimate value functions by averaging the sample returns observed. |
| **Off-policy Learning** | An RL algorithm (like Q-learning) that learns the value of the optimal policy independently of the agent's actions. |
| **On-policy Learning** | An RL algorithm (like Sarsa) that learns the value of the policy being carried out by the agent. |
| **Policy (π)** | A mapping from states to actions, defining the agent's behavior. It can be deterministic (always the same action in a state) or stochastic (a probability distribution over actions). |
| **Q-function** | Also known as the state-action value function, Q(s,a). It represents the expected total future reward for taking action a in state s and following the policy thereafter. |
| **Q-learning** | An off-policy, temporal-difference control algorithm that directly approximates the optimal Q-function, independent of the policy being followed. |
| **Reinforcement Learning (RL)** | A class of machine learning where an agent learns to interact with an environment to maximize a cumulative reward over many steps. |
| **Reward** | A numerical signal sent from the environment to the agent in response to an action. The agent's sole objective is to maximize the total reward it receives. |
| **Reward Shaping** | The process of designing a reward function, often by adding rewards for achieving subgoals, to guide the learning process. The principle is to specify what to do, not how to do it. |
| **Sarsa** | An on-policy temporal-difference control algorithm that updates the Q-value based on the quintuple: State, Action, Reward, next State, next Action. |
| **State** | A representation of the environment's current situation as perceived by the agent. |
| **Temporal-Difference (TD) Learning** | An RL method that combines ideas from Monte Carlo methods and Dynamic Programming. It learns directly from experience (like MC) but updates values based on estimates of successor states (like DP). |
| **Value Function** | A function that estimates the total expected future reward from a given state (V(s)) or from a given state-action pair (Q(s,a)). |
