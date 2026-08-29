---
LOrder: 440
layout: post
title: DecisionTree and Bagging
lecture: S4-DecisionTree
lectureVersion: current
video: <b>M1</b> <a href="https://youtu.be/JKcTiyvIpp8" target="_blank">Original</a> · <a href="https://youtu.be/Wfc-BFw2o3I" target="_blank">AI-Clone</a> · <a href="https://youtu.be/b8prnd2sFlE" target="_blank">Author-Voice</a>  |  <b>M2</b> <a href="https://youtu.be/iKTxnJU0L1E" target="_blank">Original</a> · <a href="https://youtu.be/0qhC6DoLVnE" target="_blank">AI-Clone</a> · <a href="https://youtu.be/uLDSAzmi8cg" target="_blank">Author-Voice</a>  |  <b>M3</b> <a href="https://youtu.be/WaWTw07Luzs" target="_blank">Original</a> · <a href="https://youtu.be/SQSxXbYFFTI" target="_blank">AI-Clone</a> · <a href="https://youtu.be/3lFnVcObQZk" target="_blank">Author-Voice</a>
notes: 'Daumé III (2017), <a href="http://ciml.info/dl/v0_99/ciml-v0_99-all.pdf" target="_blank">A Course in Machine Learning, Ch. 1: Decision Trees</a> · Olah (2015), <a href="https://colah.github.io/posts/2015-09-Visual-Information/" target="_blank">Visual Information Theory</a>'
categories: tabular
tags:
- 3Classification
- Ensemble
- 5Theory
---

# Summary: 

- Module1: Basic Tree4Classifier @ https://youtu.be/JKcTiyvIpp8
- Module2: How2Learn Tree https://youtu.be/iKTxnJU0L1E
- Module3: Bagged DT https://youtu.be/WaWTw07Luzs



# Study Guide for Decision Trees and Ensemble Methods

## Quiz

Answer the following questions in 2-3 sentences, based on the provided lecture material.

1. What are the three major types of classification approaches described in the lecture, and which category do decision trees belong to?
2. Describe the basic anatomy of a decision tree as explained in the "Play Tennis" example.
3. What is Entropy in the context of information theory, and what does a lower entropy value signify for a node in a decision tree?
4. Explain the concept of Information Gain (IG) and its function in the process of building a decision tree.
5. What is a primary caveat or disadvantage of using Information Gain as a purity measure for selecting splits?
6. Identify two common problems associated with decision trees and list the two general approaches mentioned for addressing overfitting.
7. Define Bootstrap Aggregation (Bagging) and explain its primary purpose.
8. How does the "instability" or "high variance" of a model like a decision tree relate to the effectiveness of Bagging?
9. What are the main characteristics of the CART (Classification and Regression Trees) algorithm?
10. What is the purpose of "pruning" in decision tree construction, and what are the two main types mentioned?

---

## Answer Key

1. The three major types of classification approaches are Discriminative, Generative, and Instance-based. Decision trees are a discriminative approach, meaning they directly estimate a decision rule or boundary to separate classes.

2. A decision tree consists of nodes, which represent tests on a specific feature or attribute. The branches from a node correspond to the possible values of that attribute, and the leaves of the tree represent the final decisions or classifications. As data is passed down the tree, the sample size at each node gets progressively smaller.

3. Entropy is defined as the expected amount of information when observing the output of a random variable. In a decision tree, entropy measures the purity of a node; a lower entropy value indicates that the distribution of classes within that node is less uniform and therefore purer.

4. Information Gain, calculated as entropy(parent) – [average entropy(children)], measures the reduction in uncertainty about the target variable (Y) after knowing a feature variable (X). It is used as a criterion to select which attribute provides the best split at each node, with the attribute offering the highest IG being chosen.

5. A primary caveat is that the number of possible values for an attribute influences the information gain. An attribute with more possible values is more likely to have a higher gain because it can more easily form small, pure partitions, which can be misleading.

6. Two common problems are the instability of trees (high variance) and overfitting. The two approaches to combat overfitting are: 1) stop growing the tree when further splitting provides no improvement, or 2) grow a full tree and then prune it by eliminating nodes.

7. Bootstrap Aggregation, or Bagging, is a technique for reducing the variance of a prediction function. It works by creating multiple bootstrap samples (datasets drawn with replacement from the original training data), fitting a model to each sample, and then having the models vote on the final prediction.

8. Model instability is considered beneficial for Bagging. High-variability methods like decision trees see more improvement from Bagging because the process of averaging predictions from different models (fit on different data samples) helps to smooth out their instability and reduce overall variance.

9. The CART algorithm is a popular tree-builder that constructs binary trees. It can work with discrete, continuous, and categorical values, handles missing values using "surrogate splits," and uses cost-complexity pruning to reduce overfitting. The Scikit-learn library uses CART for its tree implementations.

10. Pruning is a technique used to reduce the size of decision trees by removing branches with little predictive power, which helps to reduce overfitting. The two main types mentioned are Reduced Error Pruning, which replaces nodes starting from the leaves, and Cost Complexity Pruning, which removes the subtree that minimizes a specific error-to-complexity ratio.

---

## Essay Questions

Formulate comprehensive, essay-style answers to the following prompts. No answer key is provided.

1. Using the "Play Tennis" example, explain how a decision tree represents a "disjunction of conjunctions." Detail the logical structure and show how a new test case is classified by being passed down the tree.

2. Describe the role of information theory in the greedy construction of a decision tree. Define Information, Entropy, and Information Gain, explaining the mathematical relationship between them and how they guide the selection of the optimal attribute at each split.

3. Discuss the problem of overfitting in decision trees. Elaborate on why decision trees are particularly susceptible to this issue, mentioning concepts like model instability and error propagation.

4. Explain the Bagging technique in detail. Describe the process of creating bootstrap samples, training a "committee of trees," and making a final prediction. Why is this method particularly effective for high-variance models?

5. Compare and contrast the ID3, C4.5, and CART tree-building algorithms. Discuss their historical progression, the purity metrics they commonly use, and their respective capabilities in handling different data types (continuous vs. discrete), missing values, and overfitting.

---

## Glossary of Key Terms

| Term | Definition |
|------|------------|
| **Bagging (Bootstrap Aggregation)** | A technique for reducing the variance of a prediction function by creating a committee of models (e.g., trees) trained on bootstrap samples of the data and having them vote on the predicted class. |
| **Bootstrap** | The process of randomly drawing datasets with replacement from the original training data. Each new sample is the same size as the original training set. |
| **C4.5** | A tree-building algorithm and successor to ID3. It can handle both continuous and discrete features, manage missing values, and uses pruning to reduce overfitting. |
| **CART (Classification and Regression Trees)** | A popular tree-building algorithm that constructs binary trees. It can handle various value types, uses surrogate splits for missing data, and employs cost-complexity pruning. It is used by Scikit-learn. |
| **Conditional Entropy** | The entropy of a variable Y given that another variable X is known, expressed as H(Y\|X). It represents the average entropy of the children nodes after a split on variable X. |
| **Discriminative Classifier** | A type of classification approach that directly estimates a decision rule or boundary. Examples include decision trees, SVMs, and logistic regression. |
| **Entropy (H(X))** | The expected amount of information when observing the output of a random variable. In machine learning, it measures the impurity or non-uniformity of a set of examples; lower entropy signifies higher purity. |
| **Generative Classifier** | A type of classification approach that builds a generative statistical model for the data. Examples include Naïve Bayes classifiers and Bayesian networks. |
| **Gini (Population Impurity)** | An alternative purity measure to Information Gain used for splitting nodes in a decision tree. |
| **ID3 (Iterative Dichotomiser 3)** | An early tree-building algorithm that uses a top-down, greedy approach with the Information Gain metric. It only works with discrete features. |
| **Information** | A measure of the reduction in uncertainty or the amount of surprise in an outcome. It is calculated as I(X) = -log₂(p(x)). |
| **Information Gain (IG)** | The reduction in uncertainty of a target variable achieved by knowing a feature variable. Calculated as entropy(parent) – average entropy(children), it is used to select the best attribute for a split. |
| **Instance-based Classifier** | A type of classification approach that uses observations directly without building an explicit model. An example is the K-Nearest Neighbors (KNN) algorithm. |
| **Overfitting** | A phenomenon where a model learns the training data so perfectly that it captures noise and random fluctuations, leading to poor performance on new, unseen data. Decision trees are susceptible to this. |
| **Pruning** | The process of reducing the size of a decision tree by removing branches that have little predictive power. This is a technique to combat overfitting. |
| **Purity** | A measure of how uniform the class distribution is within a node. A completely pure node contains samples of only one class. |
| **Regression Tree** | A type of decision tree used for regression tasks, where the prediction is typically the average of the values at a given leaf node. It can be considered a piecewise constant regression model. |
| **Surrogate Splits** | A technique used in the CART algorithm to handle missing data. A surrogate split is a mimic of the original split in a node but uses a different predictor, allowing an observation with a missing value to proceed down the tree. |
