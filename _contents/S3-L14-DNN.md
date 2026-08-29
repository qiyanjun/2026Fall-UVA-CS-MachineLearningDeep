---
LOrder: 300
layout: post
title: Quick survey of recent deep learning
lecture: S3-deepNNSurvey
lectureVersion: current
extraContent: S3-25recentLLM-extra
video: <b>M1</b> <a href="https://youtu.be/tjIJVfwSrgs" target="_blank">Original</a> · <a href="https://youtu.be/VlRvDEgGlio" target="_blank">AI-Clone</a> · <a href="https://youtu.be/ERmB14NBze4" target="_blank">Author-Voice</a>  |  <b>M2</b> <a href="https://youtu.be/Z2Pq6QokCco" target="_blank">Original</a> · <a href="https://youtu.be/RKjZQ0sVkH8" target="_blank">AI-Clone</a> · <a href="https://youtu.be/uej_07mCf3U" target="_blank">Author-Voice</a>  |  <b>M3</b> <a href="https://youtu.be/l58pIDazkQU" target="_blank">Original</a> · <a href="https://youtu.be/akDRFnuW8I8" target="_blank">AI-Clone</a> · <a href="https://youtu.be/q36MqvNWXq8" target="_blank">Author-Voice</a>  |  <b>M4</b> <a href="https://youtu.be/gjXxOVly83s" target="_blank">Original</a> · <a href="https://youtu.be/Mo0qGBHNO3k" target="_blank">AI-Clone</a> · <a href="https://youtu.be/O4BzakYswuA" target="_blank">Author-Voice</a>
notes: Goodfellow, Bengio & Courville (2016), <a href="https://www.deeplearningbook.org/" target="_blank">Deep Learning</a> · LeCun, Bengio & Hinton (2015), <a href="https://www.nature.com/articles/nature14539" target="_blank">Deep Learning (Nature review)</a>
morenotes: <a href="https://github.com/afshinea/stanford-cs-230-deep-learning"> DNN Cheatsheets </a> + <a href="https://github.com/qiyanjun/2026Fall-UVA-CS-MachineLearningDeep/blob/main/Lectures/S3-25recentLLM-extra.pdf"> [Recent LLM-survey] </a> 
extra: true
categories: survey
tags:
- 3Classification
- Nonlinear
- Deep
- Discriminative
- 4Unsupervised
- Generative
---

### Extra reaing: 
<a href="https://github.com/qiyanjun/2026Fall-UVA-CS-MachineLearningDeep/blob/main/Lectures/S3-25recentLLM-extra.pdf"> [Recent LLM-survey] </a> 

### Summary: 

This lecture covers 10 deep learning trends that go beyond classic machine learning:


- 0. Popular CNN, RNN, Transformer models are not covered much here

- 1. DNN on graphs / trees / sets

- 2. NTM 4program induction

- 3. Deep Generative models/ DeepFake

- 4. Deep reinforcement learning

- 5 . Few-shots / Meta learning / AGI?

- 6. pretraining workflow / Autoencoder / self-supervised training

- 7. Generative Adversarial Networks (GAN) workflow

- 8. AutoML workflow / Learning to optimize /to search architecture

- 9. Validate / Evade / Test / Verify / Understand DNNs

- 10. Model Compression / Efficient Net

 

# Disclaimer: it is quite hard to make important topics of deep learning fit on a one-session schedule. We aim to make the content reasonably digestible in an introductory manner. We try to focus on a modularity view by introducing important variables to digest deep learning into chunks regarding data/ model architecture /tasks / training workflows and model characteristics. We think this teaching style provides students with context concerning those choices and helps them build a much deeper understanding.



# Briefing on Recent Trends in Deep Neural Networks

## Executive Summary

This document synthesizes an overview of ten significant trends in Deep Neural Networks (DNNs), based on a lecture from the University of Virginia's CS 4774 Machine Learning course. The analysis is structured around the "Deep Learning in a Nutshell" framework, which deconstructs the machine learning process into core components: Task, Representation (Topology), Score Function, Search/Optimization, Data, Models/Parameters, Hyperparameters/Metrics, and Hardware.

The key trends indicate that innovation in deep learning is occurring across this entire spectrum. Advancements are not limited to model architecture but also encompass the types of problems being solved, the methods used for training and optimization, and the practical challenges of deployment. The ten identified trends are:

1. **Expanding Input Types**: Moving beyond traditional grid-like data (images, sequences) to handle complex structures like graphs, trees, and sets.
2. **Symbolic Reasoning**: Employing architectures like Neural Turing Machines (NTMs) for tasks requiring program induction and manipulation of symbolic data.
3. **Deep Generative Models**: Creating realistic data for applications like image super-resolution, text-to-image synthesis, and DeepFakes, while also aiding in semi-supervised learning and handling missing data.
4. **Deep Reinforcement Learning (RL)**: Enabling agents to learn complex behaviors from raw observations by maximizing reward signals, exemplified by AlphaGo's success.
5. **Meta-Learning and AGI**: Shifting focus from narrow, task-specific AI towards "learning to learn" and the pursuit of Artificial General Intelligence (AGI), defined by autonomy and the ability to solve novel problems.
6. **Self-Supervised Pre-training**: Using autoencoders and reconstruction loss to pre-train network layers on unlabeled data, improving feature detection and mitigating training issues like vanishing gradients.
7. **Generative Adversarial Networks (GANs)**: A powerful training workflow for generative tasks, with notable advancements seen in models like CycleGAN, Progressive GAN, and StyleGAN.
8. **Automated Machine Learning (AutoML)**: Developing methods to automate the optimization process and the search for effective DNN architectures, reducing manual effort.
9. **Robustness and Trustworthiness**: A growing focus on understanding, verifying, and securing DNNs against adversarial attacks, as well as developing methods for model interpretation and bias detection.
10. **Hardware Adaptation and Efficiency**: Compressing large models for deployment on resource-constrained hardware like IoT and mobile devices, using techniques such as pruning, quantization, and filter decomposition.

---

## The "Deep Learning in a Nutshell" Framework

The lecture organizes deep learning concepts and trends within a modular framework that breaks the field into its constituent parts. This structure provides a holistic view of where innovation is occurring.

| Component | Description | Examples from Source |
|-----------|-------------|---------------------|
| **Task** | The ultimate goal or problem to be solved. | Prediction, Generation, Reinforcement, Reasoning, Classification |
| **Data (X)** | The input fed into the system. | 2D/3D Grids, 1D Grids (sequences), Graphs, Sets, Tabular Data |
| **Representation (f())** | The model's topology or architecture that transforms the data. | CNN, RNN, Transformer, GNN, NTM, Multilayer Network |
| **Score Function (L())** | A function that measures the model's performance or error. | Cross-Entropy, Mean Squared Error (MSE), Reconstruction Error |
| **Search/Optimization** | The algorithm used to find the best model parameters by minimizing the score. | Stochastic Gradient Descent (SGD), Backpropagation, Asynchronous SGD |
| **Models, Parameters** | The learnable components of the model. | Weights, Biases |
| **Hyperparameter, Metrics** | Settings that configure the learning process and metrics to evaluate success. | Network architecture choices, learning rate / Accuracy, F1 Score |
| **Hardware** | The physical infrastructure used for training and deployment. | GPU, TPU, Edge devices |

## Module I: Innovations in Representation and Architecture

This module focuses on trends related to the Representation component of the framework, specifically the types of data a DNN can process and the architectures designed for them.

### Trend 1: DNNs for Graphs, Trees, and Sets

Deep learning is expanding beyond grid-structured data. New architectures are being developed to handle more complex and irregular data formats:

- **Graph Neural Networks (GNNs)** are used for graph data.
- Other specialized topologies are designed for tree and set-based inputs.

### Trend 2: Symbolic Reasoning and Program Induction

This trend addresses tasks that require reasoning over symbolic inputs and outputs, such as computer programs.

- **Neural Turing Machines (NTMs)** are a key architecture in this area. An NTM combines a neural network controller with external memory, which it can read from and write to using "blurry" read/write heads. This enables it to handle tasks involving sequential symbolic forms and decision-making.

### Foundational Architectures

The lecture also recaps foundational architectures that remain central to the field:

- **Convolutional Neural Networks (CNNs)**: Highly effective for grid-like data. Applications mentioned include:
  - **Image Processing**: Standard use case.
  - **Playing Go**: A 19x19 game board is treated as an image to predict the next move.
  - **Speech Processing**: A spectrogram (representing frequency over time) is treated as an image, with filters moving in the frequency direction.
- **Recurrent Neural Networks (RNNs), Attention, Seq2Seq, and Transformers**: These architectures are dominant in natural language processing (NLP).
  - **Notable Pre-trained Models**: The source highlights ELMo (pre-trained biLSTM for contextual embeddings) and BERT (pre-trained transformer encoder for sentence embeddings) as significant examples.

## Module II: Expanding the Scope of Solvable Tasks

This module explores trends related to the Task component, highlighting new types of problems that deep learning is being applied to.

### Trend 3: Deep Generative Models

These models are designed to generate new, realistic data. Their applications and benefits include:

- **Applications**: Image Super-Resolution, Label-to-Image, Edges-to-Image, Text-to-Image, and DeepFakes (swapping faces in videos).
- **Utility**:
  - Testing the ability to handle high-dimensional probability distributions.
  - Simulating possible futures for planning or reinforcement learning.
  - Handling missing data and enabling semi-supervised learning.
  - Producing multi-modal outputs.

### Trend 4: Deep Reinforcement Learning (RL)

Deep RL combines deep neural networks with reinforcement learning principles.

- **Core Concept**: An agent interacts with an environment, receiving raw observations (not hand-crafted states) and a scalar reward. It learns by taking actions to maximize this cumulative reward.
- **AlphaGo Case Study**: The success of AlphaGo is attributed to a learning pipeline that combines Supervised Learning (SL) and Reinforcement Learning (RL) to guide its Monte Carlo Tree Search:
  - **SL Policy Network**: Provides prior search probability.
  - **Rollout**: Conducts quick simulations on leaf nodes of the search tree.
  - **Value Network**: Evaluates the "global feeling" or state of a leaf node.
  - Subsequent work, referenced as "Mastering the Game of Go without Human Knowledge," further advanced this approach.

### Trend 5: Meta-Learning and the Pursuit of Artificial General Intelligence (AGI)

This trend focuses on "learning to learn" and moving beyond task-specific "Narrow AI."

- **Narrow AI**: Models designed for specific tasks like game-playing, medical diagnosis, or car-driving.
- **Artificial General Intelligence (AGI)**: Defined as "the ability to achieve complex goals in complex environments using limited computational resources." Key characteristics of AGI include:
  - Autonomy.
  - Practical understanding of self and others.
  - Ability to understand "what the problem is," not just solve problems posed by programmers.
  - Capability to solve problems unknown to its original programmers.

## Module III: Evolving Training and Optimization Workflows

This module covers trends related to the Search/Optimization and Score Function components, focusing on new ways to train models effectively.

### Trend 6: Self-Supervised Learning and Autoencoders

This approach involves training models on unlabeled data by creating a supervisory signal from the data itself.

- **Autoencoders**: An auto-encoder is trained to reproduce its input. The Reconstruction Loss (difference between input and output) forces the hidden layers to learn reliable and descriptive features.
- **Unsupervised Pre-training**: A layer-wise training workflow:
  1. Train the first hidden layer using a self-supervised loss, then fix its parameters.
  2. Repeat this process for subsequent layers, working bottom-up.
  3. After pre-training, perform Supervised Fine-tuning on labeled data to refine the learned features for the specific end-task.
- **Benefits**: This method helps overcome common DNN training challenges like non-convexity and vanishing gradients.

### Trend 7: Generative Adversarial Networks (GANs)

GANs represent a specific training framework for generative models, involving a competition between a generator and a discriminator. Notable advanced GAN architectures cited include:

- **CycleGAN**: Translates characteristics between two image collections without paired examples (e.g., style transfer, object transfiguration).
- **Progressive GAN**: Improves quality, stability, and variation in generated images.
- **StyleGAN**: A generator architecture that produces high-quality images and achieves an unsupervised separation of high-level attributes (disentanglement).

### Trend 8: Automated Machine Learning (AutoML)

AutoML aims to automate aspects of the machine learning pipeline, particularly optimization and architecture design.

- **Key Focus**: "Learning to Optimize" and "Learning to Search" for optimal DNN architectures.
- **Example**: "Neural Optimizer Search with Reinforcement Learning" is cited as a method for hyperparameter search.

## Module IV: Addressing Deployment and Real-World Challenges

This final module addresses trends related to the practical deployment of DNNs, including robustness and hardware efficiency.

### Trend 9: Trust, Robustness, and Interpretability

As DNNs are deployed in critical systems, ensuring their reliability and understanding their behavior has become paramount.

- **Understanding DNNs**:
  - **Post-hoc Explanations**: Methods like feature visualization and attribution to explain model decisions after training.
  - **Inherently Interpretable Models**: Designing models that are transparent by construction.
- **Adversarial Examples (AE)**: DNNs can be fooled by adding imperceptible noise to inputs. The source provides an example where adding 0.007 * noise to an image of a "panda" causes a CNN to misclassify it as a "gibbon."
- **Verification**: Formal methods are being developed to prove properties about DNN behavior, such as the "Reluplex" SMT solver for verifying networks.

### Trend 10: Hardware-Aware DNNs and Model Compression

This trend is driven by the need to deploy powerful DNNs on resource-constrained hardware, such as the projected 20 billion connected IoT devices by 2020.

- **Goal**: Enable efficient edge inference for applications like language translation, speech recognition, and object detection on mobile and IoT devices.
- **Model Compression Methods**:
  - **(a) Model Pruning**: Removing redundant neurons, filters, or layers. Methods include magnitude-based pruning and Hessian-based pruning, often involving iterative pruning and retraining.
  - **(b) Simpler Filter Construction**: Using techniques like matrix factorization, singular value decomposition (SVD), and flattened convolutions.
  - **(c) Quantization**: Reducing the precision of weights, activations, and gradients using fixed-point formats or codebooks.
