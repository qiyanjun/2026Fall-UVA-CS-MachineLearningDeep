---
LOrder: 385
layout: post
title: Agent Evaluation and Safety
lecture:
extra: true
lectureVersion: current
extraContent:
notes: 'Zheng et al. (2023), <a href="https://arxiv.org/abs/2306.05685" target="_blank">Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena</a> · Willison (2023), <a href="https://simonwillison.net/2023/Apr/14/worst-that-can-happen/" target="_blank">Prompt Injection: What''s the Worst That Can Happen?</a>'
video:
categories: LLM/Agents
tags:
- 6LLMAgents
- Evaluation
- Safety
---

### In this lecture, we cover:
- Evaluating LLMs and agents: static benchmarks vs. task-completion / end-to-end success rate, and LLM-as-judge evaluation (using one model to score another's output)
- Pitfalls of LLM-as-judge (self-preference bias, verbosity bias) and why human evaluation still matters
- Alignment and safety basics: recap of RLHF/RLVR post-training from the "deep RL Gym" lecture, red-teaming, and prompt-injection as an agent-specific attack surface
- Practical guardrails for deployed agent systems: least-privilege tool access, human-in-the-loop approval for high-stakes actions, and logging/observability

# Study Guide: Agent Evaluation and Safety

## Quiz: Short-Answer Questions

1. What is "LLM-as-judge" evaluation, and what is one bias it is known to suffer from?
2. Why is end-to-end task-completion rate often a more meaningful agent metric than a static benchmark accuracy score?
3. What is prompt injection, and why is it a distinctly agent-relevant safety concern (beyond a plain chatbot)?
4. Name two practical guardrails for a deployed agent system and what risk each one addresses.

## Answer Key

1. LLM-as-judge uses a (typically strong) language model to score or compare the outputs of another model or agent run, in place of or alongside human raters, often used for scalable automated evaluation. It is known to suffer from self-preference bias (favoring outputs styled like its own) and verbosity bias (favoring longer answers regardless of actual quality).
2. Static benchmarks measure performance on fixed, often narrow question sets and can be gamed or may not reflect real deployment conditions. End-to-end task-completion rate measures whether the agent actually achieved the real-world goal (e.g., successfully booked the flight, fixed the bug) across the full multi-step interaction, which better reflects what users actually care about.
3. Prompt injection is an attack where untrusted content (e.g., a webpage, document, or tool output the agent reads) contains text crafted to be interpreted as new instructions, hijacking the agent's behavior. It is distinctly agent-relevant because agents routinely ingest external, attacker-influenceable content as part of tool use and then act on it (e.g., calling further tools), whereas a plain chatbot only responds with text to a single user-controlled prompt.
4. Least-privilege tool access (only granting the specific tools/permissions a given agent task actually needs) limits the damage a compromised or misbehaving agent can cause. Human-in-the-loop approval for high-stakes actions (e.g., sending money, deleting data) ensures an irreversible or costly action is checked by a person before execution, addressing the risk of an agent acting on a wrong or manipulated decision.

## Glossary of Key Terms

| Term | Definition |
|------|------------|
| **LLM-as-Judge** | Using an LLM to score or compare outputs in place of (or alongside) human evaluation. |
| **Task-Completion Rate** | The fraction of end-to-end tasks an agent actually completes successfully, as opposed to a static per-question accuracy score. |
| **Prompt Injection** | An attack embedding instructions in content an agent reads (not from the trusted user), aiming to hijack its behavior. |
| **Least-Privilege Access** | Granting an agent only the minimum tools/permissions needed for its task. |
| **Human-in-the-Loop** | Requiring human approval before an agent executes a high-stakes or irreversible action. |
