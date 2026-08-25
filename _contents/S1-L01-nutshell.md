---
layout: post
title: Machine Learning in a Nutshell
lecture: S1-nutshell
lectureVersion: current
extraContent: 
notes: <a href="https://scikit-learn.org/stable/auto_examples/classification/plot_digits_classification.html#sphx-glr-auto-examples-classification-plot-digits-classification-py">two modes running example</a> 
video:   <a href="https://youtu.be/CmIJoUoYJGk"  target="_blank">M1</a> / <a href="https://youtu.be/SboSRvUFKVM"  target="_blank">M2</a> / <a href="https://youtu.be/RA_MdNwfDOI"  target="_blank">M3</a>
categories: basics
tags:
- 1Basic
---

# Machine Learning in a Nutshell: Comprehensive Study Guide

## I. Core Concepts of Machine Learning

### A. Definition and Goal

- **Machine Learning (ML)**: A field of artificial intelligence where systems learn from data to optimize a performance criterion, aiming to generalize to unseen data.
- **Goal**: To enable computers to learn from example data or past experience without being explicitly programmed, and then apply this learned knowledge to make predictions or decisions on new, unseen data.

### B. The Machine Learning Pipeline (Nutshell Components)

- **Data**: The raw material for learning, consisting of examples/instances/samples, each with features/attributes and potentially a target/label.
- **Task**: The specific problem ML aims to solve (e.g., classification, regression, clustering).
- **Representation**: The way data and the learning function are structured (e.g., linear function, neural network).
- **Score Function (Loss/Cost Function)**: A metric used to evaluate how well the model's predictions align with the actual values. The goal is often to minimize this function.
- **Search/Optimization**: The algorithms used to find the best model parameters by minimizing the score function (e.g., gradient descent).

#### Models, Parameters, Hyperparameters, Metrics
- **Models**: The learned functions or structures.
- **Parameters**: Internal variables of the model learned from data (e.g., w, b in a linear classifier).
- **Hyperparameters**: Parameters whose values are set before the learning process begins (e.g., learning rate, regularization strength).
- **Metrics**: Measures used to evaluate model performance (e.g., accuracy, precision, recall, MSE).

### C. Two Phases of Machine Learning

#### Training (Learning Parameters)
- Uses a training set of input-output pairs (X, Y).
- The objective is to find model parameters (e.g., w, b) that minimize the loss/cost function, representing the difference between predicted f(x) and true y on the training examples.

#### Testing / Deployment / Inference (Evaluating Performance)
- Uses a testing set of examples (x?) that were not included in the training set.
- Evaluates the model's performance by measuring the difference between true y? and predicted f(x?).
- Crucial for assessing the model's ability to generalize to new data.

### D. When to Use Machine Learning

- **Extracting knowledge from data**: Discovering hidden relationships and correlations in large datasets.
- **Learning tasks difficult to formalize**: Problems hard to define explicitly with rules (e.g., face recognition).
- **Creating software that improves over time**: Systems that can adapt to new knowledge and data without continuous manual redesign.

## II. Data Types in Machine Learning

### A. General Terminology

- **Data/points/instances/examples/samples/records**: Rows in a dataset.
- **Features/attributes/dimensions/independent variables/covariates/predictors/regressors**: Columns in a dataset (excluding the target).
- **Target/outcome/response/label/dependent variable**: The special column to be predicted.

### B. Main Types of Columns

- **Continuous**: Real numbers (e.g., weight, temperature).
- **Discrete**: Symbols or categories (e.g., "Good", "Bad", "Cat", "Dog").

### C. Examples of Data Types

- **Tabular Data**: Structured data in rows and columns (e.g., patient records with gene quantities).
- **1D Sequence Data**: Data with a sequential order (e.g., language text, genome sequences, audio).
- **2D Grid Data**: Data arranged in a grid (e.g., images).
- **Graph Data**: Data representing relationships between entities (e.g., social networks).

## III. Machine Learning Tasks

### A. Supervised Learning

**Definition**: Learning from labeled input-output pairs (X, Y). The goal is to predict Y for unseen X.

#### Classification: Predicting a discrete target variable (Y)
- **Binary Classification**: Two possible output classes (e.g., positive/negative review, disease present/absent).
- **Multi-class Classification**: More than two possible output classes (e.g., categorizing text into Sports, Business, Education).
- **Multi-label Classification**: Assigning a set of target labels to a single input (e.g., an image being tagged with "Castle" and "Mountains").
- **Hierarchical Classification**: Classification where classes are organized in a hierarchy.

#### Regression: Predicting a continuous target variable (Y)

#### Generating: Creating new data instances based on learned patterns (e.g., Text2Image, Edges2Image)

### B. Unsupervised Learning

- **Definition**: Learning from unlabeled data (no Y provided).
- **Goal**: Finding patterns, structures, or relationships within the data.
- **Example**: Clustering – grouping instances into "natural" clusters.

### C. Structured Output Learning

- **Definition**: Prediction tasks where output labels have complex structured correlations or constraints (e.g., spatial, temporal, relational dependencies).
- **Example**: Language parsing (predicting a parse tree), sequence labeling (predicting a sequence of labels for a sequence of inputs).

### D. Reinforcement Learning (RL)

- **Definition**: An agent interacts with an environment and learns to maximize a scalar reward signal.
- **Key Characteristics**: Not independent and identically distributed (IID) data, sequential decision-making.
- **Variations**: Basic RL (no labels/supervision), imitation learning (supervised).

## IV. Representation Types

### A. Concept of Representation

How the input data (X) and the function (f) mapping X to Y are structured.

### B. Examples of Representations

- **Linear functions**: f(x,w,b) = sign(w^T x + b) for binary classification.
- **Nonlinear functions**: Polynomial expansion, logistic function, trees, multi-layer networks (neural networks).
- **Prob-density family**: Bernoulli, multinomial, Gaussian, mixture of Gaussians.
- **Vector Space Representation**: Representing text as vectors (e.g., Bag of Words).
- **Representation Learning (Deep Learning/Feature Learning)**: Automatically learning useful features from raw data, often through multi-layer architectures. This replaces traditional, time-consuming "Feature Engineering."

## V. Loss/Cost Types

### A. Purpose of Loss Functions

- Quantify the difference between the model's prediction and the true target value.
- Minimizing the loss function guides the learning process.

### B. Examples of Loss Functions

- **Mean Squared Error (MSE)**: Common for regression tasks.
- **Hinge Loss**: Used in binary classification (e.g., Support Vector Machines).
- **Log-likelihood / Cross-entropy**: Common for classification tasks, especially with probabilistic models.
- **0-1 Loss**: Simple loss for classification, counts misclassifications.
- **Regularized loss functions (L1, L2)**: Add penalty terms to the loss function to control model complexity and prevent overfitting.

## VI. Optimization and Model Properties

### A. Search/Optimization Algorithms

**Purpose**: To find the optimal model parameters by minimizing the loss/cost function.

**Examples**:
- **Gradient Descent (GD)**: An iterative first-order optimization algorithm that takes steps proportional to the negative of the gradient.
- **Stochastic Gradient Descent (SGD)**: A variant of GD that updates parameters using a single training example or a small batch at a time.
- **Newton's Method**: A second-order optimization algorithm.
- **Linear Programming, Quadratic Programming**: For specific types of optimization problems.
- **EM (Expectation-Maximization)**: For models with latent variables.
- **Backpropagation**: Used to train neural networks by efficiently computing gradients.

### B. Model Properties / Basic Concepts

- **Generalization**: The ability of a model to perform well on unseen data. This is the ultimate goal of ML.
- **Overfitting**: When a model learns the training data too well, including noise, leading to poor performance on new data.
- **Underfitting**: When a model is too simple to capture the underlying patterns in the data, leading to poor performance on both training and test data.
- **Bias-Variance Tradeoff**: A fundamental concept balancing model simplicity (high bias, low variance) and complexity (low bias, high variance).
  - Big gap between train/validation loss suggests overfitting (high variance).
  - No gap, both bad, suggests underfitting (high bias).
- **Regularization**: Techniques (e.g., L1, L2) that add a penalty to the loss function to discourage overly complex models and prevent overfitting.

## VII. Evaluation and Advanced Considerations

### A. Measuring Prediction Accuracy

- Evaluate models on test data using appropriate metrics.
- For supervised classification, common metrics include accuracy, precision, recall, F1-score.

### B. Human-Centric ML/AI Research

- **Trustworthy ML**: Ensuring models are reliable and fair.
- **Explainable ML**: Understanding why a model makes certain predictions.
- **Robustness**: How well a model performs under adversarial conditions (e.g., adversarial examples).
- **Interacting with/Understanding/Modeling Human Intent**: Developing AI that better collaborates with humans.

## Quiz: Machine Learning Fundamentals

**Instructions**: Answer each question in 2-3 sentences.

1. What is the primary goal of machine learning, and how does it differ from traditional programming?
2. Explain the concept of "generalization" in machine learning. Why is it important, and what happens if a model fails to generalize?
3. Describe the two main phases of a machine learning process. What is the key difference in the data used during these phases?
4. Define a "loss function" and explain its role in training a machine learning model.
5. What is the difference between "parameters" and "hyperparameters" in a machine learning model? Provide a brief example for each.
6. List three types of data commonly encountered in machine learning. Give a real-world example for each.
7. Distinguish between supervised and unsupervised learning tasks. Provide a task example for each.
8. Briefly explain what "representation learning" is and why it has become more prominent compared to traditional "feature engineering."
9. What is the "bias-variance tradeoff"? How might you diagnose a model that is suffering from high variance?
10. Describe the purpose of "regularization" in machine learning. How does it help in model training?

## Answer Key

1. **Primary Goal of ML**: The primary goal of ML is to optimize a performance criterion using example data or past experience, aiming to generalize to unseen data. It differs from traditional programming where rules are explicitly coded; instead, ML systems learn patterns from data to make predictions or decisions.

2. **Generalization Concept**: Generalization refers to a model's ability to perform accurately on new, unseen data, not just the data it was trained on. It is crucial because the ultimate purpose of an ML model is to make predictions in real-world scenarios. A model that fails to generalize is said to be overfitting, meaning it has learned the training data too specifically, including noise, and will perform poorly on new examples.

3. **Two Main Phases**: The two main phases are training and testing/deployment. During training, the model learns parameters from a labeled "training set." During testing, the model's performance is evaluated on a separate "testing set" of examples that were not part of the training data.

4. **Loss Function**: A loss function quantifies the discrepancy between a model's predicted output and the actual true output for a given example. Its role in training is to provide a metric that the optimization algorithm (e.g., gradient descent) attempts to minimize, thereby iteratively adjusting the model's parameters to improve its accuracy.

5. **Parameters vs Hyperparameters**: Parameters are internal variables of the model learned from the data during training, like the weights (w) and bias (b) in a linear classifier. Hyperparameters are external configurations set before the training process begins, such as the learning rate in gradient descent or the regularization strength.

6. **Three Data Types**: Three common data types are tabular, 2D grid, and 1D sequence data. Tabular data can be patient records with various medical attributes. 2D grid data commonly refers to images, like photographs for object recognition. 1D sequence data includes text documents for sentiment analysis or genome sequences.

7. **Supervised vs Unsupervised**: Supervised learning involves learning from labeled input-output pairs (X, Y) to predict Y for new X; an example is classifying emails as spam or not spam. Unsupervised learning, conversely, deals with unlabeled data to find patterns or structures within it; an example is clustering customer data to identify distinct market segments.

8. **Representation Learning**: Representation learning is the process where a machine learning model automatically discovers the representations (features) needed for detection or classification from raw data. It has become prominent because it often yields more effective features and reduces the laborious, time-consuming, and often task-dependent manual effort required in traditional "feature engineering."

9. **Bias-Variance Tradeoff**: The bias-variance tradeoff refers to the dilemma of simultaneously minimizing two sources of error that prevent models from generalizing well: bias (error from overly simplistic assumptions) and variance (error from sensitivity to small fluctuations in the training data). High variance (overfitting) can be diagnosed by observing a large gap between the model's performance on the training data (very good) and its performance on validation or test data (significantly worse).

10. **Regularization Purpose**: Regularization is a technique used to prevent overfitting by adding a penalty term to the loss function during training. This penalty discourages the model from becoming too complex or assigning excessively large weights to features. By controlling model complexity, regularization helps the model generalize better to unseen data.