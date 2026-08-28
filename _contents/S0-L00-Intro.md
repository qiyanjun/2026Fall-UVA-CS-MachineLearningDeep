---
LOrder: 10
layout: post
title: Introduction
lecture: S0-L1-Intro
lectureVersion: current
extraContent: 
notes: <a href="http://cs231n.github.io/python-numpy-tutorial/"> Numpy Tutorial </a>
video:  <a href="https://youtu.be/vzSWAu3tQh8"  target="_blank">M2</a> / <a href="https://youtu.be/OK1cuwZc7Rc"  target="_blank">M3</a> 
categories: basics
tags:
- 1Basic
---

# Lecture 1 Study Guide

## Course Logistics and Overview

### Course Staff and Communication
- **Instructor**: Prof. Yanjun Qi (Pronounced: /ch ee/)
- **Communication**: Email and Slack (details on Course overview)
- **TA and Office Hours**: Information available on the Course Website

### Course Materials
- **Textbook**: None required; multiple reference books shared via Course Website
- **Official Content**: Anything not mentioned in Prof. Qi's slides is not an official topic
- **Resources**:
  - All slides, videos, readings, and homework on Course Website
  - Assignments/Project: UVA CourseSite
  - Quizzes: Google Forms

### Background Needed
- **Required**: 
  - Calculus
  - Basic linear algebra
  - Basic probability
  - Basic Algorithm
- **Recommended**: Statistics
- **Programming**: Python (required for all assignments)

### Assessment Breakdown
- **Weekly Quizzes**: 15% (top 10 of 12 quiz opportunities count, 10 minutes each, closed book via Google Form; a separate Q0 PreQuiz in week 1 is an ungraded screening quiz)
- **Homework Assignments**: 50% (HW1-HW5, 10% each)
- **Midterm Exam**: 15% (October 8th)
- **Final Exam**: 20% (December 8th)
- *See CourseWeb for detailed policy*

## Course Format (2026F Flipped Classroom - Starting Week 3)

### In-Class Activities
- Pre-class question submission
- Quizzes
- Quiz review
- Assignment review
- In-class solution examples
- Short presentations/code demos

### Emphasis
- Active learning
- Understanding core ML concepts
- Applying models (Python, scikit-learn, Keras)
- Analyzing performance and ethics
- Becoming an ML tool builder

### Key Objectives
- Learn fundamental principles, algorithm design, methods, and applications
- Build simple machine learning tools from scratch (not just tool users)
- Understand complex machine learning methods at the source code level

## Machine Learning History and Concepts

### Artificial Intelligence (AI)
- **Definition**: The study of computer systems that attempt to model and apply the intelligence of the human mind
- **Attributes of Intelligence** (for machines):
  - Perceive
  - Understand
  - Interact
  - Think/Reason/Learn
  - Imagine/Make analogy
- **Impact**: Economic, cultural, social, health disruptions; potential for job automation; existential threat concerns

### History of AI/ML (Timeline)
- **1950**: Alan Turing's "Computing Machinery and Intelligence" ("Can machines think?")
- **1956**: Dartmouth workshop, John McCarthy coins "artificial intelligence"
- **1950s-60s**: Enthusiasm, early Neural Networks (NNs) popular
- **70s-80s**: Rule-based/Knowledge-based systems, Expert systems
- **1959**: Arthur Samuel popularizes "machine learning" (Samuel Checkers-playing Program)
- **90s**: Classic ML (SVM, RF, BN), Convolutional Neural Networks (CNNs) and Recurrent Neural Networks (RNNs) invented
- **2000s**: ImageNet, Deep Learning breakthroughs
- **2010s**: Deep Learning breakthroughs continue (2015-2016 CNN/RNN invented, 2017 Deep Learning, Deep Reinforcement Learning, GANs)
- **2023-Now**: ChatGPT / Generative AI / Agentic AI

### Reasons for 2010s Deep Learning Breakthroughs
1. Plenty of good quality (labeled) data (text, visual, audio, user activity, knowledge graphs, genomics, medical imaging)
2. Advanced computer architecture that fits Deep Learning (e.g., GPUs for faster, more accurate results)
3. Powerful, well-engineered machine learning libraries (easy to learn, use, extend)

### Recent Advances (2023-Now) on Generative AI
- **Definition**: AI systems that create content (text, images, audio, video, code), powered by foundation models (transformers, diffusion, LLMs). Distinct from discriminative AI
- **Key Applications**:
  - LLMs (text/chat)
  - Diffusion models (images)
  - Gen-2/Sora (video)
  - GitHub Copilot (code)
  - Multi-modal models
- **Enabling Factors**: Scaling law, data, advanced architecture, powerful libraries

## Machine Learning Basics

### Goal
Build computer systems that learn and adapt from "experience" (data examples)

### Traditional Programming vs. Machine Learning
- **Traditional**: Computer + Data + Program → Output
- **ML (Training Phase)**: Computer + Input Data (X) + Output (Y) → Program/Model f()

### Common Notations
- **Inputs (X)**: 
  - Matrix (bold capital)
  - p variables, n observations
  - Xⱼ is jth element
  - Xⱼ,ᵢ is ith observed value
  - Vectors are column vectors
- **Outputs**:
  - Quantitative (Y)
  - Qualitative/Categorical (C)
- **Observed Variables**: Lowercase

### Supervised Learning
Find a function f() to map input space X to output space Y such that the difference between true y and f(x) is small

#### Example: Linear Binary Classifier
f(x,w,b) = sign(wᵀx + b)
- wᵀx + b > 0 (e.g., +1 point)
- wᵀx + b < 0 (e.g., -1 point)

#### Training
Learning parameters w, b by minimizing a loss/cost function L() on the training set (available x and y pairs)

#### Testing
Evaluating performance on "future" (unseen) points by comparing true y to predicted f(x). Key: testing examples are not in the training set

### Basic Concepts
- **Loss Function**: Measures difference between y and f(x) (e.g., hinge loss for binary classification)
- **Regularization**: Additional information added to the loss function to control f
- **Generalization**: The ability to learn a function from past data to "explain," "predict," "model," or "control" new, unseen data examples (inductive reasoning)

### Two Modes of ML
1. **Training**: Input-output pairs (X, Y) are fed into a model to learn f()
2. **Testing/Production**: Unseen input X' is fed into the learned model f() to produce f(X') (predicted output)

### Three Aimed Features of ML Models
1. Robustness
2. Computation
3. Prediction

### General Lessons for Excellence
- Good breadth in fundamentals is key
- Strength in particular targeted topics helps stand out

### Recommended Extra-Curriculum Books
1. *Master Algorithm* by Pedro Domingos (explores different "tribes" of ML: Symbolists, Connectionists, Evolutionists, Bayesians, Analogizers)
2. *Homo Deus: A Brief History of Tomorrow* by Yuval Noah Harari

---

# Quiz: UVA CS 4774: Machine Learning - Introduction

**Instructions**: Answer each question in 2-3 sentences.

5. What is the main distinction between "Traditional Programming" and the "Machine Learning (training phase)" as illustrated in the lecture?
6. Name three specific reasons cited in the lecture for the significant breakthroughs in Deep Learning during the 2010s.
7. Define "Generative AI" and explain how it differs from "discriminative AI" as presented in the source material.
8. In the context of supervised learning, what is the purpose of the "training phase" for a linear binary classifier, and what is being minimized?
9. What is the concept of "generalization" in machine learning, and why is it crucial for model performance?

---

# Quiz Answer Key

4. **Machine Learning Goal**  
   The primary goal of machine learning is to build computer systems that can learn and adapt from their "experience." In this context, "experience" refers to the available data examples, also known as instances or samples, which are described with various properties.

5. **Traditional vs ML Programming**  
   Traditional Programming involves a computer executing a pre-defined program on data to produce an output. In contrast, the Machine Learning training phase takes input data (X) and corresponding outputs (Y) to learn or create the program/model f().

6. **Deep Learning Breakthroughs**  
   The significant breakthroughs in Deep Learning during the 2010s were primarily due to the availability of plenty of good quality labeled data, advanced computer architectures suitable for Deep Learning (like GPUs), and powerful, well-engineered machine learning libraries.

7. **Generative AI**  
   Generative AI refers to AI systems that are capable of creating new content, such as text, images, or code, powered by foundation models. This differs from discriminative AI, which primarily focuses on tasks like classification rather than content generation.

8. **Training Phase Purpose**  
   In supervised learning, the training phase for a linear binary classifier aims to learn the parameters w and b. This is achieved by minimizing a loss or cost function L(), which quantifies the difference between the true output y and the model's prediction f(x) on the available training examples.

9. **Generalization**  
   Generalization in machine learning is the ability of a model to apply the knowledge learned from past, observed data to effectively "explain," "predict," or "model" new, unseen data examples. It is crucial because it indicates whether the model has truly learned the underlying patterns rather than just memorizing the training data.

