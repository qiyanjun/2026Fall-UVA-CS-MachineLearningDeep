---
LOrder: 310
layout: post
title: Deep Learning on Text - from BOW to RNNs
lecture: S3-L16-deepNNtext-basics
lectureVersion: current
video: <b>M1</b> <a href="https://youtu.be/iTgy525nBq4" target="_blank">Original</a> · <a href="https://youtu.be/sxxi5PpV0sQ" target="_blank">AI-Clone</a> · <a href="https://youtu.be/DbtOGRrXorA" target="_blank">Author-Voice</a>  |  <b>M2</b> <a href="https://youtu.be/w22S24kFMmA" target="_blank">Original</a> · <a href="https://youtu.be/kJkAD_3qL5A" target="_blank">AI-Clone</a> · <a href="https://youtu.be/CplyalZ3wfM" target="_blank">Author-Voice</a>  |  <b>M3</b> <a href="https://youtu.be/5UGmLbAvUH0" target="_blank">Original</a> · <a href="https://youtu.be/LpRefJFoPsI" target="_blank">AI-Clone</a> · <a href="https://youtu.be/Yhbcdf5NCLc" target="_blank">Author-Voice</a>
notes: 'Jurafsky & Martin (2026), <a href="https://web.stanford.edu/~jurafsky/slp3/5.pdf" target="_blank">Speech and Language Processing, Ch. 5: Embeddings</a> · Olah (2015), <a href="https://colah.github.io/posts/2015-08-Understanding-LSTMs/" target="_blank">Understanding LSTM Networks</a>'
categories: 1D (Text)
tags:
- Nonlinear
- Deep
- 4Unsupervised
- Generative
---


### In this lecture, we cover:
- What is NLP?
  + Going beyond keyword matching: identify the structure and meaning of words, sentences, texts and conversations
  + Typical tasks: machine translation, dialog systems / question answering, sentiment / opinion analysis, text classification, language comprehension, natural language instruction
  + Challenge: language is not static (it grows and changes, e.g., cyber lingo such as LOL, G2G, IDK)
  + Y has many possible kinds: the classic NLP pipeline (part-of-speech tagging as 1D grid outputs, syntactic parsing, semantic analysis such as co-reference: is [Chris] = [Mr. Robin]?)
- Text classification and the bag of words (BOW)
  + Text classification, e.g., spam filtering: input a document D, predict a class C from {c<sub>1</sub>, ..., c<sub>L</sub>}
  + BOW: represent text as a vector of word frequencies (assumes word order is not important); the term-document matrix turns a document collection into a table
  + Common refinements (stopwords, stemming); unknown (out-of-vocabulary, OOV) words and the &lt;UNK&gt; token
  + EXTRA: tf-idf, an advanced BOW
  + Limits of BOW: removes position information and cannot (or can hardly) represent word compositions
- Word embeddings
  + One-hot vectors (length = vocabulary size, sparse, no meaning) vs. dense, learned embedding vectors (D ≪ V)
  + The lookup table E (V × D weights), `nn.Embedding`, and the shapes on a sentence batch: B × T token ids → B × T × D
  + Embedding sizes: our example (3 million), GPT-2 (38.6 million), word2vec Google News (900 million); E grows with the vocabulary, not with the sentence length
  + Word2Vec: CBOW vs. skip-gram; skip-gram training pairs from a sliding window; the skip-gram softmax with two tables E and U; negative sampling (from V to k + 1 dot products)
  + What embeddings capture (similarity, analogies) and their limits: one row per word cannot tell "mouse" the animal from "mouse" the cursor device
  + Embeddings in 2026: word, token, sentence and passage embeddings; from word vectors to retrieval (the retrieval step of RAG)
  + The workflow: pretrain → publish → load → use as is / train a new head / fine-tune all (gensim, SentenceTransformer, Hugging Face Trainer)
- Recurrent neural networks (RNNs)
  + Networks with loops that operate over variable-length sequences
  + LSTM (Hochreiter & Schmidhuber, 1997) and RNN applications (2015-20)
  + A brief intro to encoder-decoder RNNs (Seq2Seq) for machine translation

*(Seq2Seq and attention are covered in the next lecture, "Seq2Seq Models - from Encoder-Decoder RNNs to Attention.")*


# Study Guide: Deep Neural Networks for Natural Language Processing

---

## Quiz: Short-Answer Questions

**Instructions:** Provide a concise answer (2-3 sentences) for each of the following questions based on the provided course material.

1. What is Natural Language Processing (NLP), and what goal does it aim at beyond simple keyword matching?
2. What challenge does the lecture point out with the slide "language is not static"?
3. Describe text classification using spam filtering as an example.
4. What is the bag of words (BOW) representation, what assumption does it make, and what are its limits?
5. How can a model handle words in the test data that never appeared in the training data?
6. What is a one-hot vector for a word, and what are its drawbacks?
7. What is the embedding lookup table E, and what does multiplying it by a one-hot vector do?
8. A batch of B sentences, each with T token ids, goes through `nn.Embedding(V, D)`. What is the output shape, and what happens to E if the sentences get longer?
9. How many weights does E hold for V = 10,000 and D = 300? For GPT-2 and for word2vec Google News?
10. Compare CBOW and skip-gram.
11. How does skip-gram build its training data from raw text? Use "a tablespoon of apricot jam" with window m = 2.
12. What are the two tables in skip-gram, and what is the problem with the full softmax loss?
13. How does negative sampling fix that problem?
14. Count the weights: V = 50,000 and D = 100. What is the shape of E and its number of weights? A batch has 8 sentences of 20 words each: what is the shape of X? Skip-gram with both tables: how many weights? Doubling the sentence length: what happens to E and to X? With k = 10, how many dot products does one training pair need, and with the full softmax?
15. Why is "one row per word" not enough, and how do Transformers address this?
16. What is today's default kind of embedding, according to the "Embeddings in 2026" slide?
17. How are dense embeddings used for retrieval, and how does this compare with BOW / tf-idf?
18. Describe the pretrain → publish → load → decide workflow and its three options.
19. Which path: (a) searching 5,000 course notes by meaning; (b) 200 labeled tweets for 5 new categories; (c) `from_pretrained(weights, freeze=True)` then training; (d) a brand-new word given to a word2vec table?
20. What is the core characteristic of RNNs, and who invented the LSTM?
21. How does an encoder-decoder RNN perform machine translation?

---

## Answer Key

1. **NLP is a field of computer science, artificial intelligence, and computational linguistics concerned with the interactions between computers and human (natural) languages.** Its goal goes beyond keyword matching: to identify the structure and meaning of words, sentences, texts and conversations, and to reach a deep understanding of broad language.

2. **Language grows and changes over time.** For example, cyber lingo adds new terms such as LOL (laugh out loud), G2G (got to go), BFN / B4N (bye for now) and IDK (I don't know).

3. **Text classification takes a document D as input and outputs a predicted class C from {c<sub>1</sub>, ..., c<sub>L</sub>}.** In spam filtering, an email is classified as 'Spam' or 'Other' by estimating P(C = spam \| D). Other applications include classifying web pages ('Student', 'Faculty', 'Other'), news stories into topics, and movie reviews ('Favorable', 'Unfavorable', 'Neutral').

4. **BOW represents text as a vector of word frequencies, e.g., great 2, love 2, recommend 1.** It assumes word order is not important, and a collection of documents becomes a term-document matrix where X(i, j) is the frequency of word j in document i. Because it removes position information, it cannot (or can hardly) represent word compositions, so it does not apply to many NLP tasks such as translation.

5. **Train a model that includes an explicit symbol for an unknown word, &lt;UNK&gt;.** Choose a vocabulary in advance and replace out-of-vocabulary (OOV) words in the corpus with &lt;UNK&gt;; &lt;UNK&gt; is also often used to replace rare words.

6. **A one-hot vector is a binary vector whose length is the vocabulary size, with a 1 in the position of the word id and 0 elsewhere.** It does not represent word meaning (the dot product of "cat" and "dog" is 0), and it is extremely high-dimensional (over 200K English words) and extremely sparse. The solution is distributional word embedding vectors.

7. **E is a V × D table of learned weights with one row per word.** Multiplying by a one-hot vector just picks out that word's row. Every entry is a weight (V · D parameters); the rows start random and training moves them. In PyTorch this is `nn.Embedding(num_embeddings=V, embedding_dim=D)`.

8. **The output X has shape B × T × D: each token id is replaced by its row in E.** E stays V × D no matter how long the sentences are; E grows with the vocabulary, not with the sentence length.

9. **V = 10,000 and D = 300 give 3,000,000 weights.** GPT-2 (V = 50,257, D = 768) has about 38.6 million embedding weights, and word2vec Google News (V = 3,000,000, D = 300) has 900 million.

10. **CBOW predicts the current word from its context; skip-gram predicts the surrounding words given the current word.** CBOW is faster to train and predicts frequent words better. Skip-gram usually performs better than CBOW and predicts rare words better.

11. **Slide a window of size m = 2 over the text and pair each target word with every word within 2 positions.** For example, "apricot" has context words "tablespoon, of, jam", so it gives 3 real pairs (label 1); fake pairs (label 0) pair the target with random words. The sentence gives 14 real pairs; a corpus of N words gives about 2mN pairs, and no human labels are needed because the text itself supplies every real pair.

12. **Skip-gram learns two V × D tables: E (target vectors v<sub>w</sub>) and U (context vectors u<sub>c</sub>), so 2 · V · D weights (6 million in the example).** The original loss uses a softmax over the vocabulary, a V-way classifier for each pair, so its denominator sums over all V words for every pair. After training, E is kept and U is usually dropped.

13. **Negative sampling replaces the V-way softmax with k + 1 yes/no questions.** For the real pair (w, c) it pushes the dot product up, and for k random words it pushes the dot product down, using a binary cross-entropy (log-sigmoid) loss. With V = 10,000 and k = 5, one training pair needs 6 dot products (and updates 6 rows of U) instead of 10,000.

14. **E is 50,000 × 100, which is 5,000,000 weights, and X is 8 × 20 × 100.** Skip-gram with both tables has 2 · V · D = 10,000,000 weights. Doubling the sentence length leaves E unchanged and doubles T in X (8 × 40 × 100). With k = 10, one pair needs k + 1 = 11 dot products, versus 50,000 with the full softmax.

15. **Word2vec gives "mouse" the same row in "a mouse ate the cheese" and "click the mouse button", so it cannot separate the animal from the cursor device.** For word2vec the vector is the final answer. In Transformers, E is only the first layer: attention layers then mix in the context to give different vectors, and E (still V × D) is trained with the whole model.

16. **Today's default is one vector for a whole text, from a pretrained Transformer, usually a pooled summary of the T × D outputs.** Examples are all-MiniLM-L6-v2 (sentence, D = 384), BGE-M3 (passage, D = 1024) and text-embedding-3-small / large (D = 1536 / 3072). Word vectors (word2vec, GloVe) still teach the idea and are cheap.

17. **Embed N text chunks once into an N × D matrix M, embed the question as a vector q of length D, and score each chunk by cosine similarity (with unit-length vectors).** A BOW / tf-idf vector has length V and is mostly zeros and matches the same words; a dense embedding has length D, all non-zero, matches similar meaning, and needs a pretrained model. This is the retrieval step of RAG; 10,000 chunks × 384 is 3.84 million numbers, about 15 MB.

18. **Someone pretrains on huge text data, publishes weights + config on a hub (Hugging Face), and you load them with `from_pretrained(name)`.** Then you decide: use as is (no training at all), train a new head (freeze the pretrained weights), or fine-tune all (update every weight). "Fine-tuning is identical to pretraining except you don't start with random weights." The rule: start with as-is and measure, and fine-tune only if the result is not good enough.

19. **(a) Use as is: general text, no labels, search by meaning; M is 5,000 × 384 and the score vector has length 5,000.** (b) Freeze and train a new head: a few labeled examples with new labels. (c) With freeze=True, E stays fixed during training. (d) The word is not in the vocabulary, so the word2vec table has no row for it (one remedy is mapping it to &lt;UNK&gt;).

20. **RNNs are networks with loops in them, allowing information to persist, so they can operate over sequences of vectors of variable length.** Each unit uses recent history and current input, modeling dynamic temporal dependency. The LSTM ("Long short-term memory") was invented by Hochreiter & Schmidhuber (1997).

21. **An encoder RNN encodes the input sentence into a hidden state (feature), and a decoder RNN takes that hidden state as input and outputs the translated sentence.** For example, two LSTMs translate German to English: an encoder LSTM on German and a decoder LSTM on English. Seq2Seq and attention are covered in the next lecture.

---

## Essay Questions

**Instructions:** Prepare a detailed, essay-format response for each of the following prompts. (Answers not provided).

1. **Trace the evolution of text representation from bag of words through word embeddings to RNN/LSTM sequence models.** Discuss what each stage adds and what limits it (word order, meaning, context).

2. **Explain how skip-gram learns word embeddings from raw text.** Cover the training pairs, the two tables E and U, the full softmax loss and its cost, and how negative sampling reduces it, with weight and dot-product counts.

3. **Discuss the limits of static word embeddings and how embeddings are used in 2026.** Use the "mouse" example, contextual Transformer embeddings, sentence/passage embeddings, and dense retrieval for RAG.

4. **You are given a new text task. Describe how you would choose among using a pretrained model as is, training a new head, and fine-tuning all weights.** Relate the choice to the amount of labeled data, how different your text is, and your compute budget.

---

## Glossary of Key Terms

| Term | Definition |
|------|------------|
| **Natural Language Processing (NLP)** | A field of computer science, AI, and computational linguistics concerned with the interactions between computers and human (natural) languages. |
| **Text classification** | Predicting a class C from {c<sub>1</sub>, ..., c<sub>L</sub>} for an input document D, e.g., spam filtering ('Spam' vs. 'Other'). |
| **Bag of Words (BOW)** | Represents text as a vector of word frequencies, assuming word order is not important; it removes position information and cannot easily represent word compositions. |
| **Term-document matrix** | A table where entry X(i, j) is the frequency of word j in document i. |
| **tf-idf** | (EXTRA) An advanced BOW that converts word counts in the term-document matrix to tf-idf scores. |
| **OOV / &lt;UNK&gt;** | Out-of-vocabulary words are test words not seen in training; they (and often rare words) are replaced with an explicit unknown-word symbol &lt;UNK&gt;. |
| **One-hot vector** | A binary vector of vocabulary length with a 1 at the word id; extremely high-dimensional, sparse, and does not represent meaning. |
| **Embedding lookup table E** | A V × D matrix of learned weights with one row per word; `nn.Embedding` in PyTorch. Token ids of shape B × T map to embeddings of shape B × T × D. |
| **Word2Vec** | A computationally efficient, 2-layer predictive NN for learning word embeddings from raw text (Mikolov et al. 2013), with the CBOW and skip-gram models. |
| **CBOW** | A Word2Vec model that predicts the current word from its context; faster to train, better for frequent words. |
| **Skip-gram** | A Word2Vec model that predicts the surrounding words given the current word; usually better than CBOW, better for rare words. Learns two V × D tables, E (target) and U (context). |
| **Negative sampling** | Replaces the V-way softmax with k + 1 yes/no questions: push up the dot product of the real pair and push down those of k random words. |
| **Static vs. contextual embedding** | Word2vec gives one fixed vector per word (same "mouse" in every sentence); in Transformers, attention mixes in context so the same word gets different vectors. |
| **Dense retrieval** | Embed N chunks into an N × D matrix, embed the question, and rank chunks by cosine similarity; the retrieval step of RAG. |
| **Fine-tuning** | Training a pretrained model on new data; identical to pretraining except you don't start with random weights. Options: use as is, train a new head (frozen base), or fine-tune all. |
| **Recurrent Neural Network (RNN)** | A network with loops that lets information persist and operates over sequences of vectors of variable length. |
| **Long Short-Term Memory (LSTM)** | A recurrent NN invented by Hochreiter & Schmidhuber (1997); highly successful in language modeling and sequence learning. |
| **Encoder-decoder (Seq2Seq)** | An encoder RNN maps the input sentence to a hidden state, and a decoder RNN generates the output sentence from it, e.g., for machine translation. |
