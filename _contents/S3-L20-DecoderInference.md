---
LOrder: 340
layout: post
title: Transformer Inference - Decoding and Efficient Inference
lecture: S3-L20-Decoder_Inference
lectureVersion: next
extraContent:
notes: 'Holtzman et al. (2020), <a href="https://arxiv.org/abs/1904.09751" target="_blank">The Curious Case of Neural Text Degeneration</a> · Dao et al. (2022), <a href="https://arxiv.org/abs/2205.14135" target="_blank">FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness</a> · Hinton, Vinyals & Dean (2015), <a href="https://arxiv.org/abs/1503.02531" target="_blank">Distilling the Knowledge in a Neural Network</a> · Kwon et al. (2023), <a href="https://arxiv.org/abs/2309.06180" target="_blank">Efficient Memory Management for Large Language Model Serving with PagedAttention</a> · Leviathan et al. (2023), <a href="https://arxiv.org/abs/2211.17192" target="_blank">Fast Inference from Transformers via Speculative Decoding</a>'
morenotes: '<a href="https://arxiv.org/abs/2305.13245" target="_blank">Ainslie et al. (2023), GQA</a> + <a href="https://arxiv.org/abs/2405.04434" target="_blank">DeepSeek-AI (2024), DeepSeek-V2 (MLA)</a> + <a href="https://arxiv.org/abs/2309.17453" target="_blank">Xiao et al. (2024), StreamingLLM</a> + <a href="https://arxiv.org/abs/2306.14048" target="_blank">Zhang et al. (2023), H2O</a> + <a href="https://www.usenix.org/conference/osdi22/presentation/yu" target="_blank">Yu et al. (2022), Orca (continuous batching)</a>'
notebooks: '<a href="https://colab.research.google.com/github/qiyanjun/2026Fall-UVA-CS-MachineLearningDeep/blob/main/notebook/L20-DecoderInference-notebook-kv-cache-quantization-speculative.ipynb" target="_blank">notebook/L20-DecoderInference-notebook-kv-cache-quantization-speculative.ipynb</a>'
video:
categories: LLM/Agents
tags:
- 6LLMAgents
- Deep
- Generative
- Efficiency
- Deployment
---

### In this lecture, we cover:
- Recap: the decoder-only Transformer, end to end (tokenization → embedding → Transformer → linear + softmax head → pick a token → append and repeat)
  + Same trained network throughout: no loss and no gradients, only fast and cheap forward passes; search now runs over output sequences, not weights
- Generating text
  + The autoregressive loop: run the model on the tokens so far, take the logits at the last position, pick the next token, append it, and stop at &lt;eos&gt; or a length limit
  + Picking the next token: greedy (argmax), sampling, temperature τ (τ &lt; 1 sharper, τ &gt; 1 flatter), and top-k / top-p
  + Beam search vs. greedy decoding, and why randomness is needed in open-ended generation (neural text degeneration)
- Where the cost goes
  + Prefill (read the whole prompt in one pass, compute-bound) vs. decode (one token per pass, memory-bound); time to first token (TTFT), time per output token (TPOT), and throughput
  + The KV cache: causal masking means old keys and values never change, so compute q, k, v for the new token only and reuse the cache (not valid for BERT)
  + KV cache memory: 2 × layers × t × D values (≈ 2.1 GB per sequence for 32 layers, D = 4096, 4,096 tokens, 16 bits); a real model's bill (Gemma 2 27B)
  + Memory = shared weights + one cache per user; the decode speed limit set by memory bandwidth; from tokens to dollars (batching keeps the GPU busy)
- Cheaper attention
  + FlashAttention: compute scores one tile at a time in fast on-chip memory with a running (online) softmax; exact, same math, less data movement
  + Sharing K and V: multi-query attention (MQA) and grouped-query attention (GQA), e.g., Llama-2-70B with 8 KV heads instead of 64
  + Compressing K and V: multi-head latent attention (MLA, DeepSeek)
  + Changing the attention range: sliding-window attention and StreamingLLM (keep the first tokens too); pruning the KV cache (Scissorhands, H2O)
  + Mixture-of-experts: cuts compute per token, but not hosting memory
- Smaller models
  + Quantization: lower precision for weights and activations (FP16 → INT8 / INT4); post-training quantization (PTQ) vs. quantization-aware training (QAT)
  + Knowledge distillation: train a small student to match a large teacher's output distribution
  + Pruning and sparsity: remove redundant weights or attention heads with limited quality loss
  + Small language models (SLMs) and on-device deployment: what is given up, and when a smaller model is the right engineering choice
- Faster decoding and serving
  + Speculative decoding: a small draft model guesses, the large model verifies; when it pays (acceptance rate, light load, predictable text)
  + Prompt caching: reuse a shared prefix, put what changes last, and how providers price it
  + Continuous batching and PagedAttention (a block table for the KV cache, as in vLLM)
- Choosing what to do: which trick first (long wait for the first word, slow words, idle GPU, out of memory, bill too high), and measure first

*(Next: prompting, retrieval (RAG), and test-time compute in "Prompting and RAG".)*


# Study Guide: Transformer Inference - Decoding and Efficient Inference

---

## Quiz: Short-Answer Questions

**Instructions:** Provide a concise answer (2-3 sentences) for each of the following questions.

1. You run greedy decoding twice on the same prompt. Do you get the same output? What about sampling at τ = 1?
2. Logits z = [2, 1, 0]. What does softmax(z / τ) approach as τ → 0? As τ → ∞?
3. Why does beam search suit translation but give bland stories?
4. Why can a decoder reuse old keys and values, but BERT cannot?
5. What are the prefill and decode phases, and which one is limited by memory bandwidth?
6. A 10B model is stored at 8 bits on a GPU with 2 TB/s memory bandwidth. Bound its tokens per second for one user. What about at 4 bits?
7. When you add users to one GPU, which part of memory grows and which stays the same?
8. Why is FlashAttention exact, unlike GQA or a sliding window?
9. A model uses 16 heads and a 2.1 GB KV cache per user. With 4 KV heads (GQA), what does the cache become?
10. Your GPU cannot hold a model. Does switching to a mixture-of-experts (MoE) model of the same total size help?
11. How does speculative decoding speed up generation without changing the output, and why can it get slower at peak load?
12. Every request begins with the same 3,000-token prompt, except for a timestamp on line 1. What is wasted, and what is the fix?
13. What is the difference between post-training quantization and quantization-aware training?
14. How does knowledge distillation let a smaller model approach a larger model's quality?
15. Give a scenario where deploying a smaller, less capable model is the better engineering decision.

---

## Answer Key

1. **Greedy decoding always takes the argmax, so it gives the same output every time.** Sampling at τ = 1 draws from the softmax distribution, so the outputs usually differ from run to run.

2. **As τ → 0, the distribution approaches one-hot on the largest logit, [1, 0, 0], which is greedy decoding.** As τ → ∞, all logits are divided toward 0, so it approaches the uniform distribution [1/3, 1/3, 1/3].

3. **Beam search keeps the few most likely partial sequences and returns a high-probability output, which suits tasks with one mostly correct answer, such as translation.** For open-ended stories, the most likely text is generic and repetitive, so sampling (with temperature or top-p) gives more natural and varied text.

4. **In a decoder, the causal mask means old tokens never attend to new ones, so their keys and values never change and can be cached.** In BERT, attention is bidirectional, so a new token changes every old representation and the whole sequence must be recomputed.

5. **Prefill reads the whole prompt in one forward pass and is compute-bound; it sets the time to first token (TTFT).** Decode adds one token per forward pass, rereading all the weights and the KV cache each step, so it is memory-bound and sets the time per output token (TPOT).

6. **At 8 bits the weights take about 10 GB, and each decode step must read them all, so the bound is about 2 TB/s ÷ 10 GB ≈ 200 tokens/s.** At 4 bits the weights take about 5 GB, so the bound rises to about 400 tokens/s: fewer bytes to move means faster decode.

7. **The weights are shared, so they stay the same.** Each user pays for their own KV cache, so the cache grows with the number of users (and their context lengths), and it often limits the batch size.

8. **FlashAttention computes the same softmax attention, one tile at a time with a running maximum and running sum, and never stores the full score matrix.** It only reorders the work to move less data, so it needs no new quality evaluation; GQA and sliding windows change the model itself.

9. **The cache size is proportional to the number of KV heads, so 4 KV heads instead of 16 make it 4× smaller: about 2.1 / 4 ≈ 0.53 GB per user.** That leaves room for more users per GPU, or a longer context.

10. **No. MoE activates only a few experts per token, which cuts compute per token, but all experts must still be stored.** The hosting memory depends on the total number of parameters, so a MoE of the same total size still does not fit.

11. **A small draft model guesses several next tokens, and the large model checks all of them in one forward pass, keeping the accepted prefix.** When most guesses are right, several tokens come out in the time of one, and the output matches what the large model would produce. Under heavy load the GPU is already busy with batching, so the draft cost and the rejected work can outweigh the gain.

12. **The timestamp on line 1 makes every prefix different, so the 3,000 shared tokens are recomputed (prefilled) for every request and the prompt cache never hits.** Put what changes last: move the timestamp to the end so the shared prefix can be cached and reused.

13. **Post-training quantization (PTQ) converts an already-trained model's weights to lower precision after the fact; it is cheap but can lose some accuracy.** Quantization-aware training (QAT) simulates low-precision arithmetic during training or fine-tuning, so the model learns to compensate and usually keeps more accuracy at the same bit-width.

14. **The student is trained on the teacher's soft output probabilities, not only the hard labels.** These soft targets carry extra information (how confident the teacher is across several plausible answers), so the student can copy much of the teacher's behavior with far fewer parameters.

15. **A support chatbot inside a mobile app with a tight latency budget and no reliable network is better served by a small, quantized, on-device model.** It answers instantly and offline at almost no cost per query; here latency, connectivity, and cost matter more than the last few points of quality.

---

## Essay Questions

**Instructions:** Prepare a detailed, essay-format response for each of the following prompts. (Answers not provided).

1. **Walk through generating one sentence with a decoder-only Transformer.** Describe the autoregressive loop, how the next token is picked (greedy, sampling with temperature, top-k / top-p, beam search), and how the KV cache changes the cost of each step.

2. **Explain where the memory and time go when serving an LLM to many users.** Compare prefill and decode, compute the KV cache size for a model of your choice, and explain how the weights, the cache, memory bandwidth, and batching determine tokens per second and cost per million tokens.

3. **Compare the methods for making attention and serving cheaper (FlashAttention, MQA / GQA, MLA, sliding windows, StreamingLLM, KV cache pruning, speculative decoding, prompt caching, continuous batching, PagedAttention, quantization, distillation, pruning).** For each, say what it changes, whether it changes the model's outputs, and which symptom (slow first word, slow words, idle GPU, out of memory, high bill) it fixes.

---

## Glossary of Key Terms

| Term | Definition |
|------|------------|
| **Autoregressive Decoding** | Generating one token at a time: each step takes everything produced so far and predicts one more token. |
| **Greedy Decoding** | Always picking the most likely next token (argmax). |
| **Temperature (τ)** | Dividing the logits by τ before the softmax: τ &lt; 1 sharpens and τ &gt; 1 flattens the distribution. |
| **Top-k / Top-p Sampling** | Sampling only from the k most likely tokens, or from the smallest set whose total probability is at least p. |
| **Beam Search** | Keeping the B most likely partial sequences at each step instead of just one. |
| **Prefill / Decode** | The two phases of a request: reading the whole prompt in one pass, then generating one token per pass. |
| **TTFT / TPOT** | Time to first token, and time per output token. |
| **KV Cache** | Stored keys and values of earlier tokens, per layer, so each new token computes only its own q, k, v. |
| **FlashAttention** | An exact attention algorithm that works in tiles with a running softmax to reduce memory traffic. |
| **MQA / GQA** | Multi-query and grouped-query attention: several query heads share one set of key and value heads, shrinking the KV cache. |
| **Multi-head Latent Attention (MLA)** | Compressing keys and values into a smaller latent vector to shrink the KV cache. |
| **Sliding-Window Attention** | Each token attends only to the most recent W tokens. |
| **StreamingLLM** | A sliding window that also keeps the first few tokens ("attention sinks"). |
| **Quantization (PTQ / QAT)** | Storing weights and activations in fewer bits; done after training (post-training) or simulated during training (quantization-aware). |
| **Knowledge Distillation** | Training a smaller student model to mimic a larger teacher model's output distribution. |
| **Pruning** | Removing redundant weights, neurons, or attention heads from a trained model to reduce its size. |
| **Small Language Model (SLM)** | A compact language model that trades some capability for lower latency, memory, and deployment cost. |
| **Mixture-of-Experts (MoE)** | A model that activates only a few expert sub-networks per token: less compute, same hosting memory. |
| **Speculative Decoding** | A small draft model guesses several tokens, which the large model verifies in one pass. |
| **Prompt Caching** | Reusing the KV cache of a shared prompt prefix across requests. |
| **Continuous Batching** | Updating the batch after every decoding step, inserting new requests as others finish, so the GPU never idles. |
| **PagedAttention** | Storing the KV cache in fixed-size blocks tracked by a block table, as in vLLM, to avoid wasted memory. |
