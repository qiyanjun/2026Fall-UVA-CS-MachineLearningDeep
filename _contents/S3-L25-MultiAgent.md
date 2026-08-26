---
LOrder: 380
layout: post
title: Multi-Agent Systems, Evaluation and Safety
lecture:
lectureVersion: current
extraContent:
notes:
video:
categories: LLM/Agents
tags:
- 6LLMAgents
- MultiAgent
- Safety
- Evaluation
---

### In this lecture, we cover:
- Why use multiple agents: specialization, parallelism, and separating "doing" from "checking"
- Common multi-agent patterns: planner/worker (orchestrator delegates sub-tasks), debate (two agents argue to surface errors), and critique/reflection (a second pass reviews and revises the first agent's output)
- Coordination basics: shared task state, message passing between agents, and tool/resource access shared across agents
- Evaluating LLMs and agents: static benchmarks vs. task-completion / end-to-end success rate, and LLM-as-judge evaluation (using one model to score another's output)
- Pitfalls of LLM-as-judge (self-preference bias, verbosity bias) and why human evaluation still matters
- Alignment and safety basics: recap of RLHF/RLVR post-training from the "deep RL Gym" lecture, red-teaming, and prompt-injection as an agent-specific attack surface
- Practical guardrails for deployed agent systems: least-privilege tool access, human-in-the-loop approval for high-stakes actions, and logging/observability

# Study Guide: Multi-Agent Systems, Evaluation and Safety

## Quiz: Short-Answer Questions

1. What is the motivation for splitting a task across multiple specialized agents rather than using one agent with all tools and instructions?
2. Describe the planner/worker multi-agent pattern and one advantage it has over a single flat agent.
3. What is "LLM-as-judge" evaluation, and what is one bias it is known to suffer from?
4. Why is end-to-end task-completion rate often a more meaningful agent metric than a static benchmark accuracy score?
5. What is prompt injection, and why is it a distinctly agent-relevant safety concern (beyond a plain chatbot)?
6. Name two practical guardrails for a deployed agent system and what risk each one addresses.

## Answer Key

1. A single agent juggling many tools and a long, complex instruction set is more prone to confusion, context-window bloat, and cross-task interference. Splitting work across specialized agents (each with a narrower toolset and prompt) tends to keep each agent's context focused, allows work to run in parallel, and separates "doing" a task from "checking" it, which can catch errors a single self-reviewing agent would miss.
2. A planner (orchestrator) agent breaks a high-level goal into an ordered set of sub-tasks and dispatches each to a worker agent, then integrates the results. Compared to one flat agent trying to do everything, this isolates failures to individual sub-tasks, lets sub-tasks use different specialized tools/prompts, and makes the overall process easier to inspect and retry at a fine granularity.
3. LLM-as-judge uses a (typically strong) language model to score or compare the outputs of another model or agent run, in place of or alongside human raters, often used for scalable automated evaluation. It is known to suffer from self-preference bias (favoring outputs styled like its own) and verbosity bias (favoring longer answers regardless of actual quality).
4. Static benchmarks measure performance on fixed, often narrow question sets and can be gamed or may not reflect real deployment conditions. End-to-end task-completion rate measures whether the agent actually achieved the real-world goal (e.g., successfully booked the flight, fixed the bug) across the full multi-step interaction, which better reflects what users actually care about.
5. Prompt injection is an attack where untrusted content (e.g., a webpage, document, or tool output the agent reads) contains text crafted to be interpreted as new instructions, hijacking the agent's behavior. It is distinctly agent-relevant because agents routinely ingest external, attacker-influenceable content as part of tool use and then act on it (e.g., calling further tools), whereas a plain chatbot only responds with text to a single user-controlled prompt.
6. Least-privilege tool access (only granting the specific tools/permissions a given agent task actually needs) limits the damage a compromised or misbehaving agent can cause. Human-in-the-loop approval for high-stakes actions (e.g., sending money, deleting data) ensures an irreversible or costly action is checked by a person before execution, addressing the risk of an agent acting on a wrong or manipulated decision.

## Glossary of Key Terms

| Term | Definition |
|------|------------|
| **Multi-Agent System** | Multiple LLM agents, each with a role/toolset, coordinating to complete a task. |
| **Planner/Worker Pattern** | An orchestrator agent decomposes a goal into sub-tasks and delegates them to worker agents. |
| **Debate Pattern** | Two agents argue opposing positions to surface errors or weak reasoning before a final answer is chosen. |
| **Critique/Reflection** | A second pass (by the same or another agent) reviews and revises an initial output. |
| **LLM-as-Judge** | Using an LLM to score or compare outputs in place of (or alongside) human evaluation. |
| **Task-Completion Rate** | The fraction of end-to-end tasks an agent actually completes successfully, as opposed to a static per-question accuracy score. |
| **Prompt Injection** | An attack embedding instructions in content an agent reads (not from the trusted user), aiming to hijack its behavior. |
| **Least-Privilege Access** | Granting an agent only the minimum tools/permissions needed for its task. |
| **Human-in-the-Loop** | Requiring human approval before an agent executes a high-stakes or irreversible action. |
