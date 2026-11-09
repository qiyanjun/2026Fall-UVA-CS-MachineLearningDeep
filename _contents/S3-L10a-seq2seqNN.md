---
LOrder: 315
layout: post
title: Seq2Seq Models - from Encoder-Decoder RNNs to Attention
lecture: S3-L17-deepNNtext-seq2seq
lectureVersion: current
video: 
notes: 'Sutskever et al. (2014), <a href="https://arxiv.org/abs/1409.3215" target="_blank">Sequence to Sequence Learning with Neural Networks</a> · Bahdanau et al. (2014), <a href="https://arxiv.org/abs/1409.0473" target="_blank">Neural Machine Translation by Jointly Learning to Align and Translate</a> · Alammar (2018), <a href="https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/" target="_blank">Visualizing a Neural Machine Translation Model</a>'
categories: 1D (Text)
tags:
- Nonlinear
- Deep
- 4Unsupervised
- Generative
---


### In this lecture, we cover:
- Recap: Word2Vec training signals and RNNs for dynamic temporal dependency
- Variable-length input / output in natural language
- Seq2Seq (Encoder-Decoder)
  + Machine translation with two RNNs
  + Input / output patterns Seq2Seq can handle
  + NLP tasks: paraphrase, dialogue, question answering, style transfer
  + Bidirectional RNNs and ELMo contextual embeddings
- Attention
  + A weight for each input at each output timestep
  + The context vector as a weighted sum of hidden encodings
- From attention to self-attention and Transformers
  + Encoder-only, decoder-only, and encoder-decoder architectures (BERT, GPT, T5)
  + BERT as a denoising autoencoder, layer-wise pre-training, and supervised fine-tuning

*(Self-attention, Transformers, and the BERT/GPT family of pretrained models are covered in depth later in this section, in "Transformer NN Architecture" and "Transformer NN Training.")*


# Study Guide: Seq2Seq Models and Attention

---

## Quiz: Short-Answer Questions

**Instructions:** Provide a concise answer (2-3 sentences) for each of the following questions based on the provided course material.

1. Why is variable length a core challenge for modeling natural language, and which kind of network addresses it?
2. Explain the roles of the Encoder and the Decoder in a Seq2Seq model for machine translation.
3. How does the encoder-decoder structure of Seq2Seq relate to an autoencoder?
4. Name three NLP tasks beyond machine translation that Seq2Seq can handle, and give the source and target of one.
5. What does ELMo add to word representation that static Word2Vec embeddings do not provide?
6. What problem does attention address, and how is the context vector for an output timestep computed?
7. How does self-attention differ from the attention used in an RNN Seq2Seq model?
8. Name the three types of Transformer architectures and give an example model of each.

---

## Answer Key

1. **Sentences differ in length, so a model with a fixed-size input and output cannot represent them directly.** Recurrent Neural Networks handle this by repeating the same unit along the sequence, which lets them operate over variable-length inputs and outputs.

2. **The Encoder is an RNN that encodes the input sentence in the source language into a hidden state (feature).** The Decoder is an RNN that takes this hidden state as its input and outputs the translated sentence in the target language.

3. **Both use an encoder that maps the input to a hidden representation and a decoder that generates an output from it.** An autoencoder is trained to reconstruct its own input, while a Seq2Seq model generates a different target sequence, such as a translation.

4. **Seq2Seq can handle paraphrase generation, dialogue generation, question answering, and style transfer.** For example, in question answering the source is a question and the target is its answer.

5. **ELMo uses a pre-trained bidirectional LSTM, so a word's embedding depends on the sentence around it.** Two RNNs read the sequence in opposite directions, and their hidden states are combined with learned weights into a contextual embedding.

6. **A single encoder hidden state has to carry the whole input sentence, which makes long sequences hard to translate.** Attention gives each input a weight for each output timestep, and the context vector (e.g., C1) is the weighted sum of the hidden encodings, repeated for every later timestep.

7. **RNN Seq2Seq attention lets the decoder look at the encoder's hidden states, connecting two sequences.** Self-attention creates attention layers that map a sequence to itself, so each position attends to the other positions in the same sequence, and the Transformer needs no recurrence.

8. **The three types are decoder-only (e.g., GPT), encoder-only (e.g., BERT), and encoder-decoder (e.g., T5).** Decoder-only models suit generation, encoder-only models use context on both sides of a token, and encoder-decoder models pair the two.

---

## Essay Questions

**Instructions:** Prepare a detailed, essay-format response for each of the following prompts. (Answers not provided).

1. **Explain how a Seq2Seq model translates a sentence from one language to another, step by step.** Discuss what the encoder's hidden state must represent, and why this design struggles with long sentences.

2. **Trace the path from RNN Seq2Seq to attention, self-attention, and the Transformer.** For each step, describe what limitation it addresses and what the next step changes.

3. **Compare the self-supervised training signals used by Word2Vec, BERT's masked language modeling, and T5's corrupted-input training.** Explain why BERT can be described as a denoising autoencoder, and how layer-wise pre-training followed by supervised fine-tuning prepares such a model for a downstream task.

---

## Glossary of Key Terms

| Term | Definition |
|------|------------|
| **Attention Mechanism** | A technique that gives a weight to each input for each output timestep, so the model can focus on the most relevant parts of the input. |
| **BERT** | Bidirectional Encoder Representations from Transformers. A pre-trained encoder-only Transformer that learns by predicting masked tokens. |
| **Context Vector** | The weighted sum of the encoder's hidden encodings, computed using the attention weights for one output timestep. |
| **Decoder** | In a Seq2Seq model, the component that takes the encoder's hidden state as input and generates the output sequence. |
| **ELMo** | Embeddings from Language Models. Contextual word embeddings from a pre-trained bidirectional LSTM. |
| **Encoder** | In a Seq2Seq model, the component that encodes the input sentence into a hidden state (feature). |
| **Fine-tuning** | Supervised training with labeled data that refines pre-trained features through back-propagation for the end-to-end task. |
| **GPT** | Generative Pre-trained Transformer. A decoder-only Transformer architecture. |
| **Layer-wise Pre-training** | Training the hidden layers of a deep network one at a time on unlabeled data, before supervised fine-tuning. |
| **Masked Language Model** | A training task that hides input tokens and asks the model to predict them from the surrounding context. |
| **Self-Attention** | Attention that maps a sequence to itself, so each position can attend to the others in the same sequence. |
| **Seq2Seq** | An Encoder-Decoder architecture for sequence-to-sequence generation tasks such as machine translation, dialogue generation, and question answering. |
| **T5** | An encoder-decoder Transformer trained to recover text from corrupted input. |
| **Transformer** | A Seq2Seq architecture built on attention, with no recurrence. Encoder and decoder layers each combine self-attention with a feed-forward network. |
