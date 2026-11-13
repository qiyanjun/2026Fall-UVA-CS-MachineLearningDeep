---
LOrder: 420
layout: post
title: Decision Trees, Bagging, RF and Boosting
lecture: S4-DecisionTree
lectureVersion: current
extraContent: S4-DT-Ensemble
morenotes: <a href="https://github.com/qiyanjun/2026Fall-UVA-CS-MachineLearningDeep/blob/main/Lectures/S4-DecisionTree-example.pdf" target="_blank">Decision Tree worked example</a>
notes: 'Daumé III (2017), <a href="http://ciml.info/dl/v0_99/ciml-v0_99-all.pdf" target="_blank">A Course in Machine Learning, Ch. 1: Decision Trees</a> · Olah (2015), <a href="https://colah.github.io/posts/2015-09-Visual-Information/" target="_blank">Visual Information Theory</a> · Freund & Schapire (1997), <a href="https://www.face-rec.org/algorithms/Boosting-Ensemble/decision-theoretic_generalization.pdf" target="_blank">A Decision-Theoretic Generalization of On-Line Learning and an Application to Boosting</a> · Chen & Guestrin (2016), <a href="https://arxiv.org/abs/1603.02754" target="_blank">XGBoost: A Scalable Tree Boosting System</a>'
notebooks: '<a href="https://github.com/dmlc/xgboost" target="_blank">XGBoost library (dmlc/xgboost on GitHub)</a>'
video: <b>M1</b> <a href="https://youtu.be/JKcTiyvIpp8" target="_blank">Original</a> · <a href="https://youtu.be/Wfc-BFw2o3I" target="_blank">AI-Clone</a> · <a href="https://youtu.be/b8prnd2sFlE" target="_blank">Author-Voice</a>  |  <b>M2</b> <a href="https://youtu.be/iKTxnJU0L1E" target="_blank">Original</a> · <a href="https://youtu.be/0qhC6DoLVnE" target="_blank">AI-Clone</a> · <a href="https://youtu.be/uLDSAzmi8cg" target="_blank">Author-Voice</a>  |  <b>M3</b> <a href="https://youtu.be/WaWTw07Luzs" target="_blank">Original</a> · <a href="https://youtu.be/SQSxXbYFFTI" target="_blank">AI-Clone</a> · <a href="https://youtu.be/3lFnVcObQZk" target="_blank">Author-Voice</a>  |  <b>M4</b> <a href="https://youtu.be/NuxS9SycZG8" target="_blank">Original</a> · <a href="https://youtu.be/GIUESzPEXIU" target="_blank">AI-Clone</a> · <a href="https://youtu.be/RKqsbVUvL3s" target="_blank">Author-Voice</a>  |  <b>M5</b> <a href="https://youtu.be/m52bovS_eNA" target="_blank">Original</a> · <a href="https://youtu.be/Xf2QuLLYVQQ" target="_blank">AI-Clone</a> · <a href="https://youtu.be/tb0To0SMA8s" target="_blank">Author-Voice</a>  |  <b>M6</b> <a href="https://youtu.be/sRLyzg5hmuM" target="_blank">Original</a> · <a href="https://youtu.be/J7ETp2vZNHE" target="_blank">AI-Clone</a> · <a href="https://youtu.be/wjQ2J3Y3OEg" target="_blank">Author-Voice</a>  |  <b>M7</b> <a href="https://youtu.be/VwFTEW_NM4o" target="_blank">Original</a> · <a href="https://youtu.be/xS4q6LnM-tc" target="_blank">AI-Clone</a> · <a href="https://youtu.be/lf_bd9gKq8o" target="_blank">Author-Voice</a>
categories: tabular
tags:
- 3Classification
- Ensemble
- 5Theory
- Discriminative
---

# Summary: 
- Module1: Basic Tree4Classifier @ https://youtu.be/JKcTiyvIpp8
- Module2: How2Learn Tree https://youtu.be/iKTxnJU0L1E
- Module3: Bagged DT https://youtu.be/WaWTw07Luzs
- Module4: Committee of Models and Analysis https://youtu.be/NuxS9SycZG8
- Module5: Random Forest and Analysis https://youtu.be/m52bovS_eNA
- Module6: Stacking https://youtu.be/sRLyzg5hmuM
- Module7: Boosting  https://youtu.be/VwFTEW_NM4o



# Part 1 Study Guide: Decision Trees and Bagging

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

---

# Part 2 Study Guide: Random Forests, Stacking, and Boosting

This guide provides a comprehensive review of ensemble methods in machine learning, including Bagging, Random Forests, Stacking, and Boosting, based on the provided lecture materials. It includes a quiz to test your knowledge, an answer key, a set of essay questions for deeper consideration, and a glossary of key terms.

---

## Short-Answer Quiz

**Instructions:** Answer the following questions in 2-3 sentences each, based on the provided source material.

1. What is the core two-step framework of an ensemble method in machine learning?
2. Explain the process of "bootstrap sampling" as it is used in Bagging.
3. Why is model instability considered a beneficial trait for the base classifiers used in Bagging?
4. What is the primary limitation of Bagging that the Random Forest algorithm is designed to address?
5. How does the Random Forest algorithm de-correlate the individual decision trees within its ensemble?
6. Explain the role that pairwise correlation (ρ) between trees plays in determining the overall variance of an ensemble model.
7. Describe the general architecture of Stacking, identifying its main components.
8. What are the three core principles that guide boosting strategies?
9. Contrast the training processes of Boosting and Bagging, highlighting a key difference in their approach to the training data.
10. List at least three key features of the XGBoost implementation that contribute to its efficiency.

---

## Answer Key

1. The framework of an ensemble method involves two main steps. First, a set of diverse classifiers is generated. Second, the predictions of these individual classifiers are aggregated, for instance by taking a majority vote, to form a final prediction.

2. Bootstrap sampling is a technique of randomly drawing datasets with replacement from the original training data. Each new sample is the same size as the original training set, and because the sampling is done with replacement, individual data points can be duplicated within a sample.

3. Model instability is beneficial in Bagging because the more variable or unstable a basic model is, the more improvement can be gained by averaging multiple versions of it. High-variability models like decision trees see more improvement from Bagging than low-variability methods.

4. The primary limitation of Bagging is that the bagged trees are often correlated with each other. Random Forest is an extension of Bagging that was specifically designed to reduce this correlation between the trees.

5. Random Forest de-correlates its trees by introducing an additional layer of randomness during the tree-building process. At each node, when choosing a feature for a split, the algorithm only considers a random subset of m features from the total p available features, rather than all of them.

6. Pairwise correlation (ρ) between classifiers directly impacts the variance of the ensemble's average prediction. Even as the number of classifiers (B) approaches infinity, the variance term related to correlation (ρσ²) remains, meaning that higher correlation prevents the total variance from being minimized.

7. Stacking involves learning and combining multiple classifiers. It uses a set of "base learners" trained on the data, and their predictions are then fed as input into a "meta learner" or "blender," which is a higher-level classifier that makes the final prediction.

8. The three core principles of boosting are: (1) using many base classifiers to vote on a decision, (2) sequentially training classifiers where each corrects the mistakes of the previous ones, thereby focusing on hard examples, and (3) giving higher weight to the better-performing base classifiers.

9. Unlike Bagging, which trains classifiers in parallel on different bootstrap samples of the data, Boosting trains classifiers sequentially on the entire training set. Boosting adaptively re-weights the training examples at each step to focus on instances that previous classifiers misclassified.

10. Key features of the XGBoost implementation include: the ability to use L1 or L2 regularization, a sparsity-aware split finding algorithm for handling sparse data, and a block structure system design that enables parallel learning across multiple CPU cores. Other features include cache awareness and out-of-core computing for very large datasets.

---

## Essay Questions

**Instructions:** The following questions are designed for longer, more detailed responses. Formulate your answers by synthesizing information from across the lecture materials.

1. Compare and contrast the Bagging and Boosting ensemble methods. Discuss their approaches to training data, model creation (parallel vs. sequential), and their primary goals in improving model performance.

2. Explain the Bias-Variance Tradeoff in the context of ensemble methods. How do techniques like Bagging leverage complex, high-variance base models (like full decision trees) to create a final classifier with significantly lower variance?

3. Describe the evolution from a simple Bagged Decision Tree to a Random Forest. What specific mechanism was introduced in Random Forest, and why is it mathematically significant for reducing the ensemble's variance, as explained by the role of pairwise tree correlation?

4. Detail the sequential nature of boosting algorithms like AdaBoost. How does the process focus on "hard examples" from one iteration to the next, and what roles do example weights and classifier weights play in constructing the final, powerful model?

5. Discuss the practical implementation and features of modern boosting libraries like XGBoost and LGBM. What technical innovations allow them to be so efficient and popular in machine learning competitions, and what is a key difference in their tree growth strategies?

---

## Glossary of Key Terms

| Term | Definition |
|------|------------|
| **AdaBoost** | An adaptive boosting algorithm that sequentially trains base classifiers, focusing on hard examples from previous iterations by updating example weights, and combines them into a final weighted sum. |
| **Bagging** | Stands for "Bootstrap Aggregation." An ensemble technique for reducing the variance of a prediction function by generating multiple versions of a predictor from bootstrap samples and aggregating them. |
| **Base Learner** | An individual classifier (e.g., a decision tree) that is part of a larger ensemble model, particularly used in the context of Stacking. |
| **Boosting** | An ensemble method that sequentially trains base classifiers, where each new classifier is trained to correct the mistakes of the previous ones. It combines a weighted sum of many classifiers, often shallow decision trees. |
| **Bootstrap Sampling** | A re-sampling technique that involves randomly drawing datasets with replacement from the original training data. Each sample is the same size as the original set. |
| **Discriminative Classifier** | A type of classification approach that directly estimates a decision rule or boundary. Examples include SVM, decision trees, and neural networks. |
| **Ensemble Method** | A machine learning framework that involves generating a set of diverse classifiers and aggregating their predictions to produce a final result. Examples include Bagging, Boosting, and Stacking. |
| **Generative Classifier** | A type of classification approach that builds a generative statistical model. Examples include Bayesian networks and Naïve Bayes classifiers. |
| **Instance-based Classifier** | A type of classification approach that uses observations directly without building an explicit model. An example is the K-nearest neighbors algorithm. |
| **LGBM** | Stands for Light Gradient Boosted Machines. An efficient library for training GBMs developed by Microsoft that uses leaf-wise tree growth and novel techniques like Gradient-based one-side sampling. |
| **Meta Learner (Blender)** | In Stacking, the higher-level classifier that is trained on the predictions generated by the base learners to make the final prediction. |
| **Random Forest** | An ensemble classifier that extends Bagging by de-correlating the trees. It does this by considering only a random subset of features at each split when building the individual decision trees. |
| **Stacking** | An ensemble method where multiple base learners are trained, and a meta learner is then trained on the predictions of the base learners to produce the final output. |
| **XGBoost** | Stands for Extreme Gradient Boosting. A highly efficient and popular implementation of a Gradient Boosting Decision Tree that incorporates features like regularization, parallel learning, and handling of sparse data. |
