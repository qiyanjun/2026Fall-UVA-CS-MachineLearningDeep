---
LOrder: 360
layout: post
title: Efficient LLMs and Deployment
lecture:
lectureVersion: current
extraContent:
notes:
video:
categories: LLM/Agents
tags:
- 6LLMAgents
- Deep
- Efficiency
- Deployment
---

### In this lecture, we cover:
- Why inference efficiency matters: latency, memory footprint, and serving cost at scale (distinct from training cost)
- Quantization: reducing numeric precision of weights/activations (e.g., FP16 -> INT8/INT4), and the difference between post-training quantization and quantization-aware training
- Knowledge distillation: training a smaller "student" model to mimic a larger "teacher" model's outputs
- Pruning and sparsity: removing redundant weights or attention heads with limited quality loss
- The KV-cache: why autoregressive decoding caches past keys/values, and why that cache dominates serving memory at long context lengths
- Serving-system techniques: batching requests together, and paged/chunked KV-cache management (as in vLLM-style serving) to use GPU memory efficiently
- Speculative decoding: using a small "draft" model to propose several tokens that the large model verifies in parallel, reducing the number of expensive large-model forward passes
- Small Language Models (SLMs) and on-device deployment: what capabilities are typically sacrificed, and when a smaller model is the right engineering choice

# Study Guide: Efficient LLMs and Deployment

## Quiz: Short-Answer Questions

1. Why can a model that trains fine on a research cluster still be impractical to deploy, and what costs specifically matter at inference time?
2. What is quantization, and what is the difference between post-training quantization and quantization-aware training?
3. How does knowledge distillation let a smaller model approach a larger model's quality?
4. What is the KV-cache, why is it needed for autoregressive generation, and why does it become a memory bottleneck at long context lengths?
5. Explain the core idea behind speculative decoding, and why it can speed up generation without changing the final output distribution.
6. Give a concrete scenario where deploying a smaller, less capable model is the better engineering decision than deploying the largest available model.

## Answer Key

1. Training cost is a one-time (or infrequent) expense, but every inference call at deployment repeats the model's compute and memory cost, multiplied across potentially millions of user requests. What matters at inference time is latency (how fast a response streams back to a user), memory footprint (whether the model and its running state fit on the serving hardware), and throughput/cost (how many requests per second per GPU, which sets the dollar cost per query).
2. Quantization reduces the numerical precision used to store and compute with a model's weights (and often activations), e.g. from 16-bit floating point down to 8-bit or 4-bit integers, shrinking memory footprint and often speeding up compute. Post-training quantization converts an already-trained full-precision model's weights after the fact, which is cheap but can lose some accuracy; quantization-aware training simulates the lower-precision arithmetic during training or fine-tuning, so the model adapts to and compensates for the precision loss, typically preserving more accuracy at a given bit-width.
3. In distillation, the smaller "student" model is trained not just on hard ground-truth labels but on the larger "teacher" model's output distribution (soft probabilities) over possible answers, which carries richer information (e.g., relative confidence across multiple plausible answers) than a single hard label. Matching the teacher's distribution lets the student approximate much of the teacher's learned behavior with far fewer parameters than training the student from labels alone.
4. During autoregressive generation, computing attention for a new token requires the keys and values of all previous tokens; recomputing them from scratch at every step would be extremely wasteful, so the KV-cache stores each previous token's key/value vectors once and reuses them at every subsequent step. This cache grows linearly with sequence length (and with number of concurrent requests), and at long context lengths or high concurrency it can require more GPU memory than the model's own weights, making it the dominant memory cost in serving.
5. Speculative decoding runs a small, fast "draft" model to propose a short run of several candidate next tokens, then has the large "target" model verify all of them in a single parallel forward pass instead of one token at a time; whenever the target model's own distribution agrees with a proposed token it's accepted, and generation only falls back to the target model's own (slower) token-by-token choice at the first point of disagreement. Because accepted tokens are always ones the large model would itself have chosen (or resampled equivalently when rejected), the method is mathematically guaranteed to sample from the same distribution as running the large model alone — it just does so with fewer expensive large-model forward passes.
6. A customer-support chatbot embedded in a mobile app with a tight latency budget and no reliable network connection would be better served by a small, quantized, on-device model that responds instantly offline, even if it's less capable than a frontier cloud model, because the deployment constraints (latency, connectivity, cost per query at high volume) matter more than squeezing out the last few points of quality.

## Glossary of Key Terms

| Term | Definition |
|------|------------|
| **Quantization** | Reducing the numeric precision (bit-width) used to represent a model's weights/activations to save memory and compute. |
| **Post-Training Quantization (PTQ)** | Quantizing an already-trained model's weights after training, without further gradient updates. |
| **Quantization-Aware Training (QAT)** | Training or fine-tuning a model while simulating low-precision arithmetic, so it adapts to the resulting precision loss. |
| **Knowledge Distillation** | Training a smaller student model to mimic a larger teacher model's output distribution. |
| **Pruning** | Removing redundant weights, neurons, or attention heads from a trained model to reduce its size. |
| **KV-Cache** | Stored key/value vectors from previous tokens, reused at each autoregressive decoding step to avoid recomputation. |
| **Speculative Decoding** | Using a small draft model to propose multiple tokens verified in parallel by a large target model, reducing expensive large-model forward passes. |
| **Throughput** | The number of requests (or tokens) a serving system can process per unit time. |
| **Small Language Model (SLM)** | A compact language model designed to trade some capability for lower latency, memory footprint, and deployment cost. |
