---
LOrder: 330
layout: post
title: Transformer NN Architecture
lecture: S3-L18-TransformerArch
lectureVersion: current
extraContent:
notes: Vaswani et al. (2017), <a href="https://arxiv.org/abs/1706.03762" target="_blank">Attention Is All You Need</a> · Alammar (2018), <a href="https://jalammar.github.io/illustrated-transformer/" target="_blank">The Illustrated Transformer</a>
# morenotes: <a href="https://colab.research.google.com/github/qiyanjun/2026Fall-UVA-CS-MachineLearningDeep/blob/main/notebook/L17_Predicting_Movie_Reviews_with_BERT_PyTorch.ipynb">PyTorch Notebook on BERT for Text </a> + <a href="https://github.com/qiyanjun/2026Fall-UVA-CS-MachineLearningDeep/blob/main/Lectures/S3-L0-25recentLLM-extra.pdf"> [Recent LLM-survey] </a>
morenotes: <a href="https://github.com/qiyanjun/2026Fall-UVA-CS-MachineLearningDeep/blob/main/Lectures/S3-L0-25recentLLM-extra.pdf"> [Recent LLM-survey] </a>
video:
categories: LLM/Agents
tags:
- 6LLMAgents
- Deep
- Generative
- Architecture
- Training
notebooks: '<a href="https://colab.research.google.com/github/qiyanjun/2026Fall-UVA-CS-MachineLearningDeep/blob/main/notebook/L17_Predicting_Movie_Reviews_with_BERT_PyTorch.ipynb" target="_blank">notebook/L17_Predicting_Movie_Reviews_with_BERT_PyTorch.ipynb</a> · <a href="https://colab.research.google.com/github/qiyanjun/2026Fall-UVA-CS-MachineLearningDeep/blob/main/notebook/L18-LLMArchTraining-notebook-gpt-components-by-hand.ipynb" target="_blank">notebook/L18-LLMArchTraining-notebook-gpt-components-by-hand.ipynb</a> · <a href="https://colab.research.google.com/github/qiyanjun/2026Fall-UVA-CS-MachineLearningDeep/blob/main/notebook/L18-LLMArchTraining-notebook-gpt-components-by-hand-solutions.ipynb" target="_blank">notebook/L18-LLMArchTraining-notebook-gpt-components-by-hand-solutions.ipynb</a>'
---


### In this lecture, we cover:
- Recap: word vectors, the attention trick in Seq2Seq, and why bag-of-words fails
  + In BoW + logistic regression, "not" only adds its own weight and cannot flip the meaning of "good"
  + Goal: context-dependent word vectors built from every word in the sentence, with learned attention weights
- Self-attention basics
  + Query, key, and value: q = x W<sub>q</sub>, k = x W<sub>k</sub>, v = x W<sub>v</sub>
  + Attention scores, softmax, and the output as a weighted sum of values
  + Matrix form, one token per row (as in PyTorch): C = softmax(Q K<sup>T</sup> / √D) V
  + Why we divide by √D, and how to read an attention heatmap
- Transformer block basics
  + Multi-head self-attention: different heads for different types of relevance, combined by W<sub>O</sub>
  + Self-attention has no sense of word order, so we add positional encodings (hand-crafted or learned)
  + Learned position embeddings as a lookup table P (max_len × D, e.g. `nn.Embedding(max_len, D)`): position i reads row i, so the input is x<sub>i</sub> + p<sub>i</sub>; BERT has 512 rows, GPT-2 has 1024, and there is no row for positions ≥ max_len
  + Absolute positional embeddings: swapping two words changes their inputs, so the outputs change too
  + Residual connections, layer norm, and the feed-forward (FC) sublayer
  + Counting the parameters of one layer
- Masks and model families
  + Encoder-only (BERT): the key-padding mask (in three lines of PyTorch on a toy batch of B = 2 sentences, L = 5), masked-token pretraining, and adapting to classification with [CLS] or mean pooling
  + Autoregressive decoding: predicting one token at a time from START until an END token
  + The causal mask, combined causal + padding masks, and masked scaled dot-product attention in code
  + Encoder-decoder (Transformer, T5): cross-attention from decoder queries to encoder keys and values
  + Decoder-only (GPT): masked self-attention, no encoder and no cross-attention, trained with next-token cross-entropy

*(The roadmap also lists pretraining a GPT, generating text, and post-training. Pretraining and post-training are covered in the next two lectures, "Transformer Pretraining" and "Transformer Post-training"; generating text, the KV cache, and fast inference are covered in "Transformer Inference - Decoding and Efficient Inference". For RL post-training in more depth, see the extra "Reinforcement Learning" lecture.)*


# Study Guide: Transformer Basic Architectures

---

## Quiz: Short-Answer Questions

**Instructions:** Provide a concise answer (2-3 sentences) for each of the following questions based on the provided course material.

1. Why can a bag-of-words logistic regression not tell "the movie was good" from "the movie was not good"?
2. In self-attention, how are the query, key, and value vectors for a token computed, and which parts are learned?
3. Write the matrix form of self-attention and give the shape of each matrix for a sequence of T tokens with dimension D.
4. Why are attention scores divided by √D before the softmax?
5. What does multi-head self-attention add over a single head?
6. Why does a Transformer need positional encodings? What is the limit of a learned position-embedding table?
7. Name two ways BERT can turn its encoder outputs into a single vector for classification.
8. What is a key-padding mask, and why must masks be applied before the softmax rather than after? Write the PyTorch lines that apply it.
9. What does the causal mask do, and why is it needed for a decoder?
10. In a 6 × 6 causal mask, how many entries are −∞?
11. How does cross-attention differ from self-attention in an encoder-decoder Transformer?
12. How does a decoder-only model like GPT differ from the original Transformer decoder?

---

## Answer Key

1. **In BoW, each word adds its own fixed weight to the score, whatever the other words are.** "not" can only add its own (small) weight, so it cannot flip the meaning of "good". Bigram features such as "not good" help, but the vocabulary explodes.

2. **Each token's input vector x<sub>t</sub> is multiplied by three matrices: q<sub>t</sub> = x<sub>t</sub> W<sub>q</sub>, k<sub>t</sub> = x<sub>t</sub> W<sub>k</sub>, v<sub>t</sub> = x<sub>t</sub> W<sub>v</sub>.** The matrices W<sub>q</sub>, W<sub>k</sub>, and W<sub>v</sub> (each D × D) are the only learned parameters of self-attention.

3. **C = softmax(Q K<sup>T</sup> / √D) V, with the softmax over each row.** X, Q, K, V, and C are T × D. The score matrix S = Q K<sup>T</sup> / √D and the attention matrix A are T × T, and each row of A sums to 1. With a batch, every tensor gets a leading B.

4. **If q and k have D independent entries with mean 0 and variance 1, q · k has variance D, so scores grow like √D.** Large scores push the softmax toward a one-hot vector, gradients through the softmax almost vanish, and training stalls. Dividing by √D brings the score variance back to 1.

5. **Each head has its own query, key, and value projections, so different heads can capture different types of relevance between tokens.** Each of h heads uses D/h dimensions, and the head outputs are combined with an output matrix W<sub>O</sub>.

6. **Self-attention has no position information, so on its own it treats the input as an unordered set of tokens.** A positional vector p<sub>i</sub>, either hand-crafted or learned from data, is added to each input x<sub>i</sub> so that each position is unique. A learned table has max_len rows (512 in BERT, 1024 in GPT-2), so it has no vector for positions ≥ max_len.

7. **(A) The [CLS] token: BERT puts [CLS] at the start of every input, and its final output vector represents the whole sequence; a small linear + softmax head is added and everything is fine-tuned with cross-entropy.** (B) Mean pooling: average the output vectors over the real tokens, using a masked mean (h * mask).sum(1) / mask.sum(1) so the pads are skipped.

8. **Sequences in a batch are right-padded to the same length, and a key-padding mask stops real tokens from attending to &lt;pad&gt; by setting those score columns to −∞, since exp(−∞) = 0.** Masking after the softmax would leave rows that no longer sum to 1. In PyTorch: `key_mask = mask.unsqueeze(1)` (shape B × 1 × L, broadcast down every query row), `scores = scores.masked_fill(key_mask == 0, float("-inf"))`, then `attn = torch.softmax(scores, dim=-1)`.

9. **The causal mask lets query position t look only at keys i ≤ t, by setting the scores of later positions to −∞.** At generation time, position t + 1 does not exist yet, so the model must be trained without seeing the future.

10. **15.** The −∞ entries are the ones above the diagonal: 6 × 5 / 2 = 15.

11. **In self-attention, queries, keys, and values all come from the same sequence.** In cross-attention, the queries come from the decoder and the keys and values come from the encoder outputs. It is not causal and uses the encoder's padding mask. It is the attention trick from Seq2Seq, without an RNN.

12. **GPT keeps only the decoder: masked (causal) self-attention, with no encoder and no cross-attention.** It uses learned position embeddings and is trained with next-token cross-entropy loss at every position.

---

## Essay Questions

**Instructions:** Prepare a detailed, essay-format response for each of the following prompts. (Answers not provided).

1. **Walk through self-attention step by step for the sentence "the movie was not good".** Describe how the queries, keys, and values are computed, how the attention matrix is formed and normalized, and how the new vector for "good" can come to mean "not good".

2. **Describe the parts of one Transformer block (multi-head self-attention, positional encoding, residual connections, layer norm, and the feed-forward sublayer).** Explain what problem each part addresses, and count the parameters of one layer with D = 64.

3. **Compare the three Transformer families: encoder-only (BERT), encoder-decoder (Transformer, T5), and decoder-only (GPT).** For each, describe which kinds of attention and masks it uses, its training task, and the applications it suits.

---

## Glossary of Key Terms

| Term | Definition |
|------|------------|
| **Attention Matrix (A)** | The T × T matrix softmax(Q K<sup>T</sup> / √D). Row t shows how much output t takes from each position i, and each row sums to 1. |
| **Autoregressive Decoding** | Generating one token at a time, where each step takes everything produced so far and predicts one more token, until an END token. |
| **BERT** | An encoder-only Transformer, pretrained by predicting masked tokens (about 15% masked) and then fine-tuned to classify or tag. |
| **Causal Mask** | A mask that lets query position t attend only to keys i ≤ t, by setting later scores to −∞ before the softmax. |
| **[CLS] Token** | A special token BERT puts at the start of every input. Its final output vector represents the whole sequence for classification. |
| **Cross-Attention** | Attention in which decoder queries attend to the encoder's keys and values. |
| **Decoder-only Transformer** | A Transformer with masked self-attention, no encoder, and no cross-attention, such as GPT, trained with next-token cross-entropy. |
| **Key-Padding Mask** | A mask that sets the score columns of &lt;pad&gt; tokens to −∞ so no real token attends to padding. |
| **Layer Norm** | Normalizing each vector by subtracting its mean and dividing by its standard deviation, followed by a learned scale and shift. |
| **Mean Pooling** | Averaging output vectors over the real tokens (a masked mean that skips pads) to get one vector for the sequence. |
| **Multi-head Self-attention** | Running several attention heads in parallel, each with its own projections and D/h dimensions, and combining them with W<sub>O</sub>. |
| **Positional Encoding** | A vector p<sub>i</sub>, hand-crafted (e.g., sinusoidal) or learned, added to each input so self-attention knows token positions. |
| **Learned Position Embedding** | A trainable lookup table P with max_len rows × D columns; position i reads row i. Rows start random and are trained by backprop. |
| **Query, Key, Value** | The three projections of each token, x W<sub>q</sub>, x W<sub>k</sub>, x W<sub>v</sub>. Query-key dot products give the scores, and values are mixed by the attention weights. |
| **Residual Connection** | Adding a sublayer's input to its output (x + c) before the layer norm. |
| **Scaled Dot-Product Attention** | C = softmax(Q K<sup>T</sup> / √D) V. Dividing by √D keeps the score variance near 1. |
| **Self-Attention** | Attention that maps a sequence to itself, so each token's new vector is a weighted sum of the value vectors of all tokens. |
