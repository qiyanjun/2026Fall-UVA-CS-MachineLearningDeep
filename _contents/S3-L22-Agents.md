---
layout: post
title: Agents and Tool Use
lecture:
lectureVersion: current
extraContent:
notes:
video:
categories: LLM/Agents
tags:
- 6LLMAgents
- Agents
- Generative
---

### In this lecture, we cover:
- What makes an "agent" more than a chatbot: an LLM wrapped in a loop with tools, memory, and a goal
- Function / tool calling: how an LLM requests a structured call to an external function, and how results are fed back into context
- The ReAct pattern (Reason + Act): interleaving reasoning traces with tool actions and observations
- Planning and task decomposition: breaking a high-level goal into ordered sub-tasks, with re-planning when a step fails
- Memory: short-term (context window) vs. long-term (external stores such as vector databases or scratchpad files)
- Common agent failure modes: hallucinated tool calls/arguments, infinite tool-use loops, compounding errors across steps, and context-window budget pressure
- Basic guardrails: constraining the tool surface, validating tool arguments, and setting step/iteration limits

# Study Guide: Agents and Tool Use

## Quiz: Short-Answer Questions

1. What distinguishes an LLM "agent" from a single-turn LLM chatbot call?
2. Describe the mechanics of function/tool calling: what does the model actually output, and who executes the function?
3. What is the ReAct pattern, and why interleave reasoning text with actions instead of just calling tools directly?
4. Give an example of why an agent might need both short-term and long-term memory, and how they differ.
5. Name two common agent failure modes and a mitigation for each.
6. Why do agent loops typically need an explicit maximum step/iteration limit?

## Answer Key

1. A chatbot call maps one input to one output in a single forward pass. An agent instead runs a loop: it can decide to call external tools, observe the tool's results, and use those observations to decide its next action — repeating until it judges the task complete, which lets it interact with and change state in the outside world rather than just producing text.
2. The model is given a list of available tools (name, description, and a schema of expected arguments). Instead of producing plain text, it can output a structured request naming a tool and filling in arguments matching that schema. The calling program (not the model) actually executes the function with those arguments and returns the result, which is appended back into the model's context for its next step.
3. ReAct asks the model to alternate between short natural-language reasoning ("Thought") and concrete tool actions ("Action"), followed by the tool's result ("Observation"). Writing out the reasoning before acting tends to produce more accurate action choices and makes the agent's decision process interpretable and debuggable, compared to jumping straight to tool calls with no visible rationale.
4. An agent debugging code might need short-term memory to track the current file and error message within one session (kept in the context window), and long-term memory to recall a coding convention or prior decision from many sessions ago (stored externally, e.g. in a vector database or a persistent notes file, and retrieved when relevant since it can't all fit in one context window).
5. Hallucinated tool calls (the model invents a tool or arguments that don't match the real schema) are mitigated by strict schema validation and rejecting/reporting malformed calls. Infinite or looping tool use (the agent repeats the same failing action) is mitigated by tracking recent action history and detecting repetition, plus a hard iteration limit.
6. Without a step limit, a stuck agent (e.g., repeatedly retrying a failing action, or oscillating between two states) can loop indefinitely, burning compute/cost and never returning control to the user; a maximum iteration count guarantees the loop terminates and surfaces a failure instead of hanging forever.

## Glossary of Key Terms

| Term | Definition |
|------|------------|
| **Agent** | An LLM operating in a loop, choosing actions (including tool calls) based on observations, toward a goal. |
| **Tool / Function Calling** | A model output format requesting execution of an external function with structured arguments, executed by the surrounding program rather than the model itself. |
| **ReAct** | A prompting pattern interleaving Thought (reasoning), Action (tool call), and Observation (tool result) steps. |
| **Planning** | Decomposing a high-level goal into an ordered sequence of sub-tasks or actions. |
| **Short-Term Memory** | Information held within the current context window during a single agent run. |
| **Long-Term Memory** | Information persisted outside the context window (e.g., a vector store or file) and retrieved when relevant. |
| **Guardrail** | A constraint (schema validation, allowed tool list, iteration limit) that bounds an agent's possible actions. |
