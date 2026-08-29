---
LOrder: 340
layout: post
title: Multimodal and Vision-Language Models
lecture:
lectureVersion: current
extraContent:
notes: 'Radford et al. (2021), <a href="https://arxiv.org/abs/2103.00020" target="_blank">Learning Transferable Visual Models From Natural Language Supervision</a> · Dosovitskiy et al. (2021), <a href="https://arxiv.org/abs/2010.11929" target="_blank">An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale</a> · Liu et al. (2023), <a href="https://arxiv.org/abs/2304.08485" target="_blank">Visual Instruction Tuning</a>'
video:
categories: LLM/Agents
tags:
- 6LLMAgents
- Deep
- Generative
- Multimodal
---

### In this lecture, we cover:
- Why go beyond text: grounding language in images (and other modalities) for tasks pure-text LLMs can't do
- CLIP: contrastive image-text pretraining, and how it learns a shared embedding space where matching image/caption pairs land close together
- Turning images into "tokens": patchifying an image (as in a Vision Transformer) so a sequence model can consume it like text
- Vision-language model (VLM) architectures: a frozen or fine-tuned vision encoder feeding visual tokens into an LLM alongside text tokens (LLaVA-style), vs. natively-trained unified multimodal models (GPT-4V/Gemini-style)
- Text-to-image generation at a glance: how a text prompt conditions an image generator via cross-attention (bridges to diffusion models)
- Evaluating multimodal models: visual question answering, grounding (pointing to the right region), chart/document understanding
- Applications: multimodal RAG, and agents that can "see" (e.g., taking a screenshot as an observation)

# Study Guide: Multimodal and Vision-Language Models

## Quiz: Short-Answer Questions

1. What does CLIP's contrastive objective actually optimize, and what does the resulting shared embedding space enable?
2. How does a Vision Transformer turn an image into something a Transformer-style model can process, and how is this similar to tokenizing text?
3. Describe the LLaVA-style approach to building a vision-language model: what pieces are combined, and what's new about the design versus a plain LLM?
4. What is the difference between "bolting a vision encoder onto a frozen LLM" and training a natively multimodal model end to end?
5. Give an example of a task that plain visual question answering accuracy doesn't capture well, and explain why grounding matters there.
6. How could an LLM agent use a VLM's vision capability as a "tool" in an agent loop (recall the Agents and Tool Use lecture)?

## Answer Key

1. CLIP is trained on (image, caption) pairs with a contrastive loss: it pulls the embedding of an image and its matching caption close together while pushing away embeddings of mismatched image-caption pairs in the same batch. The resulting shared embedding space lets you compare images and text directly (e.g., by cosine similarity), enabling zero-shot image classification and image/text retrieval without task-specific training.
2. A Vision Transformer splits an image into a grid of fixed-size patches (e.g., 16x16 pixels), flattens and linearly projects each patch into a vector, and treats the resulting sequence of patch vectors as "tokens" with added positional information, exactly as a text Transformer treats a sequence of word/subword token embeddings — the same self-attention machinery then applies uniformly to both.
3. LLaVA-style models combine a pretrained vision encoder (often a CLIP-style ViT) with a pretrained LLM, connected by a small trainable projection layer that maps visual patch embeddings into the LLM's embedding space; the LLM then attends over these projected visual tokens interleaved with text tokens exactly as it would over any other tokens. What's new is mainly this lightweight adapter plus instruction-tuning on image-text conversational data, rather than training vision understanding from scratch.
4. Bolting a vision encoder onto a frozen LLM (with only a small adapter trained) is cheap and fast, reusing the LLM's existing language ability, but the LLM's representations were never optimized jointly with visual input, which can limit deep visual reasoning. Training a natively multimodal model end to end (all components co-trained, sometimes with a unified tokenizer for both modalities) is far more compute-intensive but can produce tighter integration between visual and language understanding.
5. A model can answer "how many people are in this image?" correctly by chance/statistical shortcut without actually attending to the right region of the image. Grounding — requiring the model to point to or highlight the specific pixels/region supporting its answer — checks whether the model's answer is actually tied to the correct visual evidence, not just a plausible-sounding guess.
6. A VLM can serve as a "vision" tool: the agent calls it with a screenshot or image as input and receives back a text description, an extracted value, or a grounded coordinate (e.g., "the submit button is at position (x,y)"), which the agent's core reasoning loop can then use as an observation to decide its next action — extending the ReAct-style loop from text-only tool outputs to visual ones.

## Glossary of Key Terms

| Term | Definition |
|------|------------|
| **CLIP** | A model trained with a contrastive objective on (image, caption) pairs to learn a shared image-text embedding space. |
| **Contrastive Learning** | Training that pulls matching pairs' embeddings together and pushes non-matching pairs' embeddings apart. |
| **Vision Transformer (ViT)** | A Transformer applied to images by splitting them into fixed-size patches treated as a token sequence. |
| **Patchify** | Splitting an image into a grid of fixed-size patches as a preprocessing step for a Vision Transformer. |
| **Vision-Language Model (VLM)** | A model that jointly processes image and text inputs, e.g. an LLM extended with visual tokens from a vision encoder. |
| **Adapter / Projection Layer** | A small trainable module mapping one model's representations (e.g., a vision encoder's) into another's (e.g., an LLM's) embedding space. |
| **Grounding** | Tying a model's answer to specific, verifiable evidence in the input (e.g., a region of an image). |
| **Visual Question Answering (VQA)** | The task of answering a natural-language question about an image's content. |
