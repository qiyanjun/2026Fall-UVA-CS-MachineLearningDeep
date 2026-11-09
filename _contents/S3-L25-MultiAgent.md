---
LOrder: 380
layout: post
title: Multi-Agent Systems
lecture:
extra: true
lectureVersion: current
extraContent:
notes: 'Du et al. (2023), <a href="https://arxiv.org/abs/2305.14325" target="_blank">Improving Factuality and Reasoning in Language Models through Multiagent Debate</a>'
video:
categories: LLM/Agents
tags:
- 6LLMAgents
- MultiAgent
---

### In this lecture, we cover:
- Why use multiple agents: specialization, parallelism, and separating "doing" from "checking"
- Common multi-agent patterns: planner/worker (orchestrator delegates sub-tasks), debate (two agents argue to surface errors), and critique/reflection (a second pass reviews and revises the first agent's output)
- Coordination basics: shared task state, message passing between agents, and tool/resource access shared across agents

*(Evaluating agents and agent safety are covered in the "Agent Evaluation and Safety" lecture.)*

# Study Guide: Multi-Agent Systems

## Quiz: Short-Answer Questions

1. What is the motivation for splitting a task across multiple specialized agents rather than using one agent with all tools and instructions?
2. Describe the planner/worker multi-agent pattern and one advantage it has over a single flat agent.

## Answer Key

1. A single agent juggling many tools and a long, complex instruction set is more prone to confusion, context-window bloat, and cross-task interference. Splitting work across specialized agents (each with a narrower toolset and prompt) tends to keep each agent's context focused, allows work to run in parallel, and separates "doing" a task from "checking" it, which can catch errors a single self-reviewing agent would miss.
2. A planner (orchestrator) agent breaks a high-level goal into an ordered set of sub-tasks and dispatches each to a worker agent, then integrates the results. Compared to one flat agent trying to do everything, this isolates failures to individual sub-tasks, lets sub-tasks use different specialized tools/prompts, and makes the overall process easier to inspect and retry at a fine granularity.

## Glossary of Key Terms

| Term | Definition |
|------|------------|
| **Multi-Agent System** | Multiple LLM agents, each with a role/toolset, coordinating to complete a task. |
| **Planner/Worker Pattern** | An orchestrator agent decomposes a goal into sub-tasks and delegates them to worker agents. |
| **Debate Pattern** | Two agents argue opposing positions to surface errors or weak reasoning before a final answer is chosen. |
| **Critique/Reflection** | A second pass (by the same or another agent) reviews and revises an initial output. |
