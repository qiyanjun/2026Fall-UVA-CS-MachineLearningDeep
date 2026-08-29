---
LOrder: 530
layout: post
title: Review for Final exam
lecture: S6-review
lectureVersion: next
video: <b>Video</b> <a href="https://youtu.be/fz5ExB0XP5E" target="_blank">Original</a> · <a href="https://youtu.be/GuirQX7zSuk" target="_blank">AI-Clone</a>
notes:  review for final exam + extra- <a href="https://github.com/afshinea/stanford-cs-229-machine-learning/tree/master/en">[ML Cheatsheets]</a> 
background: true
tags:
- Background
---


The source material presents a final review lecture from a university machine learning course, UVA CS 4774, which stresses the primary objective of enabling students to **build machine learning tools** from scratch, rather than merely functioning as users. The curriculum is conceptually structured around modular components, reviewing fundamental aspects such as different **data types**, various **representation methods**, applicable **score functions**, and necessary **search/optimization** strategies. It covers a wide scope of **ML tasks**, including supervised regression and classification, deep learning across different dimensional data, and advanced topics in **learning theory** and model selection. A significant portion of the review is dedicated to contextualizing the field by outlining the **Five Tribes of Machine Learning**: Symbolists, Connectionists, Evolutionists, Bayesians, and Analogizers. This framework clarifies the philosophical **origins** of each tribe and links them to their key algorithms and fundamental focus areas, such as the Connectionists' reliance on **Backpropagation** for credit assignment or the Bayesians' use of **Probabilistic inference**.


# Machine Learning Concepts and Tribes: A Study Guide

## Short-Answer Quiz

1. What is the fundamental difference in the process flow between traditional programming and machine learning?
2. According to the "Machine Learning in a Nutshell" slide, what are the three core components that define a machine learning approach?
3. List the three major types of classification approaches and briefly describe the core principle of each.
4. What are the key learning objectives for students in the UVA CS 4774 course, as stated in the review?
5. What is the bias-variance tradeoff, and what are two suggested remedies for addressing overfitting or underfitting?
6. Identify the five "tribes" of machine learning as presented by Dr. Pedro Domingos.
7. What are the origins and the key algorithm associated with the Connectionist tribe of machine learning?
8. Who termed the phrase "Bayesian network" and in what year? What is the basis for updating information in such a network?
9. Name the key algorithm for the Analogizer tribe and provide an example of a system that falls under this category.
10. Who invented Convolutional Neural Networks (CNNs), and what was a significant achievement of this invention in the 1990s?

## Quiz Answer Key

1. In traditional programming, a computer takes data and an algorithm as input to produce an output. In machine learning, the computer takes data and desired output as input, and the machine learning process itself generates the algorithm or model.
2. The three core components of machine learning are Representation (models and parameters), Score Function (a performance criterion to optimize), and Search/Optimization (the method used to optimize the score function).
3. The three major classification types are Discriminative, which directly estimates a decision boundary (e.g., logistic regression); Generative, which builds a statistical model of the data (e.g., naïve Bayes); and Instance-based, which uses training observations directly without a model (e.g., K nearest neighbors).
4. The key results for students are to be able to build a few simple machine learning methods from scratch and to understand a few complex machine learning methods at the source code and equation level. The overall objective is to enable students to build machine learning tools, not just use them.
5. The bias-variance tradeoff relates to the problems of underfitting and overfitting a model. Remedies include controlling or adjusting the model's complexity/capacity and controlling or adjusting the size of the training data set.
6. The five tribes of machine learning are Symbolists, Connectionists, Evolutionists, Bayesians, and Analogizers.
7. The Connectionist tribe has its origins in neuroscience. Its key algorithm is backpropagation.
8. The term "Bayesian network" was coined by Judea Pearl in 1985. The basis for updating information within the graph is Bayes' conditioning.
9. The key algorithm for the Analogizer tribe is "Kernel machines." Recommender Systems, which can utilize techniques like Matrix Factorization, are given as an example.
10. Professor Yann LeCun invented Convolutional Neural Networks (CNNs) in 1998. It was the first neural network to be successfully trained with many layers.

## Essay Questions

1. Compare and contrast the Symbolist and Connectionist tribes of machine learning. Discuss their philosophical origins, key algorithms, primary focus, and the historical figures associated with each movement.
2. Explain the machine learning workflow from the modular perspective presented in the course. Detail the role of each module (Data, Representation, Score Function, Search/Optimization, etc.) and provide specific examples for each from the "What we have covered" table.
3. Discuss the problem of model selection and evaluation in machine learning. Elaborate on the concepts of K-folds cross-validation, expected prediction error, and the bias-variance tradeoff as described in the source material.
4. Using the "Big Picture" table as a guide, analyze how the Bayesian and Analogizer tribes approach machine learning. Describe their origins, focus, and how their respective solutions map to the core modules of Representation and Score Function.
5. Trace the historical development of two different machine learning algorithms mentioned in the text (e.g., Decision Trees, CNNs, SVMs, or Bayesian Networks). Include the key figures, approximate timeframes, and the core contributions of each development.

## Glossary of Key Terms

| Term | Definition |
|------|------------|
| Adaboost | A decision tree-building algorithm developed by Robert Schapire in 1999. |
| Analogizers | One of the five tribes of machine learning, originating from psychology. They focus on similarity and use Kernel machines as their key algorithm. |
| Backpropagation | The key algorithm of the Connectionist tribe of machine learning. |
| Bayesians | One of the five tribes of machine learning, originating from statistics. They focus on uncertainty and use probabilistic inference as their key algorithm. |
| Bayesian Network | A probabilistic graphical model. The term was coined by Judea Pearl in 1985, using Bayes' conditioning to update information. |
| Bias and Variance Tradeoff | A central problem in machine learning concerning the balance between a model's error from erroneous assumptions (bias) and its error from sensitivity to small fluctuations in the training set (variance), leading to overfit or underfit. |
| C4.5 | A decision tree-building algorithm and successor to ID3, developed by Ross Quinlan in 1993. |
| Classification | A supervised learning task where the output variable Y is a discrete value. |
| Connectionists | One of the five tribes of machine learning, originating from neuroscience. They aim to emulate the brain and use backpropagation as their key algorithm. |
| Convolutional Neural Networks (CNN) | A class of deep neural networks, invented by Yann LeCun in 1998, notable for being the first neural network successfully trained with many layers. |
| Discriminative Classifier | A type of classification approach that directly estimates a decision rule or boundary. Examples include logistic regression and neural networks. |
| Evolutionists | One of the five tribes of machine learning, originating from evolutionary biology. They focus on structure discovery and use genetic programming as their key algorithm. |
| Generative Classifier | A type of classification approach that builds a generative statistical model. Examples include naïve Bayes classifiers and Bayesian networks. |
| Genetic Programming | The key algorithm of the Evolutionist tribe of machine learning. |
| ID3 (Iterative Dichotomiser 3) | A decision tree-building algorithm developed by Ross Quinlan in the 1980s. |
| Instance Based Classifiers | A type of classification approach that uses training observations directly without building a model. An example is K nearest neighbors. |
| Inverse Deduction | The key algorithm of the Symbolist tribe of machine learning. |
| Kernel Machines | The key algorithm of the Analogizer tribe of machine learning. The Support Vector Machine (SVM) is an important example. |
| K-folds Cross Validation | A technique used for model selection and evaluation. |
| Machine Learning | A field that grew out of AI, which aims to optimize a performance criterion using example data or past experience with the goal of generalizing to unseen data. It generates an algorithm/model from data and desired outputs. |
| Probabilistic Inference | The key algorithm of the Bayesian tribe of machine learning, used for reasoning with uncertainty. |
| Recommender Systems | An application of machine learning, often associated with the Analogizer tribe, that predicts user preferences. Matrix Factorization is an example technique. |
| Regression | A supervised learning task where the output variable Y is a continuous value. |
| Reinforcement Learning | A machine learning task focused on learning how to interact with an environment. |
| Support Vector Machine (SVM) | An important example of "kernel methods," first introduced in 1992 and popular for its success in handwritten digit recognition. |
| Symbolists | One of the five tribes of machine learning, originating from logic and philosophy. They focus on filling gaps in existing knowledge (knowledge composition) and use inverse deduction as their key algorithm. |
| Unsupervised Models | A type of machine learning task where there is no output variable Y. Examples include clustering and dimension reduction. |
