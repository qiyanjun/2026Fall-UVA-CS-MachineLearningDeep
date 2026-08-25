---
layout: post
title: Recent deep learning on Text
lecture: S3-deepNNtext
lectureVersion: current
video: <a href="https://youtu.be/iTgy525nBq4">M1</a> + <a href="https://youtu.be/w22S24kFMmA">M2</a> +   <a href="https://youtu.be/5UGmLbAvUH0">M3</a>  
notes: <a href="https://github.com/qiyanjun/2026Fall-UVA-CS-MachineLearningDeep/blob/main/notebook/L15_Predicting_Movie_Reviews_with_BERT_on_TF_Hub.ipynb">Keras Notebook on DNN text </a> + <a href="https://github.com/qiyanjun/2026Fall-UVA-CS-MachineLearningDeep/blob/main/Lectures/S3-25recentLLM-extra.pdf"> [Recent LLM-survey] </a> 
categories: 1D (Text)
tags:
- Nonlinear
- Deep
- Discriminative
- 4Unsupervised
- Generative
---


- Notebook to try: 

<a href="https://github.com/qiyanjun/2026Fall-UVA-CS-MachineLearningDeep/blob/main/notebook/L15_Predicting_Movie_Reviews_with_BERT_on_TF_Hub.ipynb">L15 Keras Notebook on DNN text </a> 

- Extra reading: 
<a href="https://github.com/qiyanjun/2026Fall-UVA-CS-MachineLearningDeep/blob/main/Lectures/S3-25recentLLM-extra.pdf"> [Recent LLM-survey] </a> 

### In this lecture, we cover: 
- What is NLP?
- Typical NLP tasks / Challenges / Pipeline
- f() on natural language
  + Before Deep NLP (Pre 2012) • (BOW / LSI / Topic Modeling LDA )
  + Word2Vec (2013-2016) • (GloVe/ FastText)
  + Recurrent NN (2014-2016) • LSTM
  + Seq2Seq
  + Attention 
  + Self-Attention (2016 – now )
  + Transformer (attention only Seq2Seq)
  + BERT / RoBERTa/ XLNet/ GPT-2 / ...



# Study Guide: Deep Neural Networks for Natural Language Processing

---

## Quiz: Short-Answer Questions

**Instructions:** Provide a concise answer (2-3 sentences) for each of the following questions based on the provided course material.

1. What is Natural Language Processing (NLP) and what fundamental goal does it aim to achieve beyond simple keyword matching?
2. Identify and briefly describe three major challenges that researchers and engineers face in the field of Natural Language Processing.
3. Explain the "Bag of Words" (BOW) representation and identify its two primary limitations for many NLP tasks.
4. What is the "one-hot vector" method for representing words, and what are its main drawbacks?
5. Describe the core characteristic of Recurrent Neural Networks (RNNs) and explain how this feature makes them suitable for processing sequence data.
6. Explain the purpose of the Encoder and Decoder components within a Seq2Seq architecture for a task like machine translation.
7. What is the core idea behind the "Attention Mechanism" in sequence-to-sequence models?
8. How does the Transformer model's architecture fundamentally differ from that of a Recurrent Neural Network?
9. Describe the Masked Language Model (MLM) pre-training objective used by BERT.
10. Differentiate between the CBOW and SkipGram models within the Word2Vec framework.

---

## Answer Key

1. **Natural Language Processing (NLP) is a field of computer science, AI, and computational linguistics focused on the interaction between computers and human languages.** Its goal is to go beyond simple keyword matching to identify the structure and meaning of words and sentences, enabling a deep understanding of broad language.

2. **Three major challenges in NLP are ambiguity, such as pronoun references; the fact that language is not static and constantly changes with new slang or "cyber lingo"; and the immense scale of language data, with sources like Wikipedia containing billions of words.**

3. **The Bag of Words (BOW) is a method that represents text by counting the occurrences of each word, such as f()=c great 2 love 2.** Its primary limitations are that it removes all word position information and cannot effectively represent word compositions.

4. **The "one-hot vector" is a binary vector with a length equal to the vocabulary size, where a '1' is placed in the position corresponding to a word's ID and the rest are '0's.** Its main drawbacks are its extremely high dimensionality, its sparsity, and its inability to represent a word's meaning.

5. **Recurrent Neural Networks (RNNs) are networks containing loops, which allow information to persist.** This structure enables them to operate over sequences of vectors with variable lengths, using recent history and current input to model dynamic temporal dependencies.

6. **In a Seq2Seq architecture, the Encoder is an RNN that encodes an input sentence (e.g., in a source language) into a hidden state or feature vector.** The Decoder is another RNN that takes this hidden state as input and generates the output sequence (e.g., the translated sentence).

7. **The Attention Mechanism provides a weight for each input word for every single output timestep.** This allows the model to create a context vector (C1) that is a weighted sum of the hidden encodings from the input, effectively letting the model focus on the most relevant parts of the input sequence when generating an output.

8. **The Transformer model's architecture is fundamentally different because it contains no recurrence.** Instead of processing sequences step-by-step like an RNN, it relies entirely on self-attention mechanisms to map a sequence to itself.

9. **The Masked Language Model (MLM) is a pre-training objective where some input tokens are masked with a unique [MASK] token.** The model is then trained as a denoising autoencoder to predict these original masked tokens based on their surrounding context.

10. **In Word2Vec, the Continuous Bag-of-Words (CBOW) model predicts the current input token based on its surrounding context tokens.** Conversely, the SkipGram model does the opposite, predicting the surrounding context tokens based on the current input token.

---

## Essay Questions

**Instructions:** Prepare a detailed, essay-format response for each of the following prompts. (Answers not provided).

1. **Trace the evolution of natural language representation in machine learning as outlined in the course material, beginning with pre-2012 methods like Bag of Words and culminating in modern approaches like Transformer-based contextual embeddings.** Discuss the key innovations and limitations at each major stage (BOW, Word2Vec, RNN/LSTM, Transformers).

2. **Compare and contrast the architectural philosophies of Recurrent Neural Networks (like LSTM) and Transformer models for processing sequential data.** What are the fundamental differences in how they handle sequences, and what specific advantages did the introduction of self-attention in Transformers provide over recurrence?

3. **Discuss the primary challenges inherent in Natural Language Processing, specifically ambiguity, scale, and the dynamic nature of language.** Using examples from the source, explain how modern deep learning approaches attempt to address these challenges more effectively than classic NLP pipeline components.

4. **Explain the concept of the Seq2Seq (Encoder-Decoder) architecture and its wide range of applications in generative NLP tasks.** How does the integration of an attention mechanism enhance the performance and interpretability of these models, particularly in a complex task like machine translation?

5. **The source mentions several landmark pre-trained models, including BERT, ALBERT, and XLNet.** Describe the concept of pre-training and fine-tuning. Explain the specific training innovations introduced by ALBERT (Sentence Order Prediction) and XLNet (Permutation Language Model) as attempts to improve upon the original BERT framework.

---

## Glossary of Key Terms

| Term | Definition |
|------|------------|
| **ALBERT** | A "lite" version of BERT that proposes the Sentence Order Prediction (SOP) task to replace Next Sentence Prediction (NSP), making the model focus more on the semantic coherence between sentences. |
| **Attention Mechanism** | A technique used in sequence models where, for each output timestep, a weighted sum of the hidden encodings of the input sequence is calculated. This allows the model to focus on the most relevant parts of the input. |
| **Bag of Words (BOW)** | A text representation method that removes word position information and represents a document as a collection of its word counts. It is not applicable to many NLP tasks because it cannot represent word compositions. |
| **BERT** | Bidirectional Encoder Representations from Transformers. A pre-trained model for sentence embedding whose architecture is a Transformer's encoder stack. It is trained using a Masked Language Model (MLM) objective. |
| **CBOW (Continuous Bag-of-Words)** | A Word2Vec model that predicts an input token based on its surrounding context tokens. |
| **Co-reference Resolution** | An NLP task that involves determining if different expressions in a text refer to the same entity (e.g., determining if "Chris" and "Mr. Robin" are the same person). |
| **Decoder** | In a Seq2Seq model, the component (typically an RNN) that takes the hidden state from the encoder as input and generates the output sequence. |
| **Encoder** | In a Seq2Seq model, the component (typically an RNN) that processes the input sentence and encodes it into a single hidden state or feature vector. |
| **GPT (Generative Pre-Training)** | A pre-trained model from OpenAI. The GPT-2 version has 1.5 billion parameters and was trained on millions of web pages. |
| **Long Short-Term Memory (LSTM)** | A type of Recurrent Neural Network (RNN) invented by Schmidhuber in 1997. It is highly successful in language modeling and sequence learning problems. |
| **Masked Language Model (MLM)** | A pre-training objective used by BERT, where some input tokens are masked and the model is trained to predict the original tokens based on their context. It functions as a Denoising Auto Encoder. |
| **Natural Language Processing (NLP)** | A field of computer science, AI, and linguistics concerned with the interactions between computers and human languages, aiming for a deep understanding of language structure and meaning. |
| **One-hot vector** | A basic method for representing a word as a binary vector whose length is the size of the vocabulary. It has a '1' in the position of the word's ID and '0's elsewhere, but it is extremely high-dimensional and sparse. |
| **Permutation Language Model (PLM)** | The pre-training objective for XLNet, which learns bidirectional contexts by permuting the factorization order of the sequence. |
| **Recurrent Neural Network (RNN)** | A type of neural network with loops, allowing information to persist. This architecture allows RNNs to operate over sequences of vectors with variable length. |
| **Self-Attention** | An attention mechanism that creates attention layers mapping from a sequence to itself, forming the core component of the Transformer model. |
| **Sentence Order Prediction (SOP)** | A pre-training task used by ALBERT where the model must determine if two sentences are in their correct original order, which forces it to learn about semantic coherence. |
| **Seq2Seq** | An Encoder-Decoder architecture used for sequence-to-sequence generation tasks like machine translation, dialogue generation, and question answering. |
| **SkipGram** | A Word2Vec model that predicts context tokens based on a given input token. |
| **Transformer** | A sequence model from Google Brain that contains no recurrence and relies entirely on self-attention mechanisms. It is a Seq2Seq model that uses encoder self-attention, decoder self-attention, and encoder-decoder attention. |
| **Word2Vec** | A technique to learn distributed representations of words (word embeddings). It includes the CBOW and SkipGram models. |
| **XLNet** | A pre-trained language model that builds on Transformer-XL (adding recurrence between segments) and uses a Permutation Language Model to learn bidirectional contexts. |

