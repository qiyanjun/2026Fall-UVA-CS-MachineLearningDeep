---
LOrder: 70
layout: post
title: GD and SGD for LR
lecture: S1-LROptimization
lectureVersion: current
extraContent:
morenotes: <a href="https://arxiv.org/abs/1609.04747"> more SGD </a> + <a href="https://web.stanford.edu/~hastie/ElemStatLearn/">ELS Ch3.2</a> + <a href="https://nbviewer.jupyter.org/github/dtnewman/gradient_descent/blob/master/stochastic_gradient_descent.ipynb"> SGD Jupyter notebook </a> + <a href="https://numpy.org/doc/stable/reference/routines.linalg.html"> numpy linalg </a>
notes: Léon Bottou (2010), <a href="https://leon.bottou.org/publications/pdf/compstat-2010.pdf" target="_blank">Large-Scale Machine Learning with Stochastic Gradient Descent</a> · Sebastian Ruder (2016), <a href="https://www.ruder.io/optimizing-gradient-descent/" target="_blank">An Overview of Gradient Descent Optimization Algorithms</a>
video: <b>M1</b> <a href="https://youtu.be/QkZIpjvtQic" target="_blank">Original</a> · <a href="https://youtu.be/QK34fBvrRaU" target="_blank">AI-Clone</a> · <a href="https://youtu.be/4OUhTeldrIs" target="_blank">Author-Voice</a>  |  <b>M2</b> <a href="https://youtu.be/7hDLwyRAjIQ" target="_blank">Original</a> · <a href="https://youtu.be/h5L2poPBcFI" target="_blank">AI-Clone</a> · <a href="https://youtu.be/x4ySfJl6L3Y" target="_blank">Author-Voice</a>
categories: tabular
tags:
- 2Regression
- Optimization
notebooks: '<a href="/2026Fall-UVA-CS-MachineLearningDeep//notebook/L4_stochastic_gradient_descent.ipynb" target="_blank">notebook/L4_stochastic_gradient_descent.ipynb</a>'
---


# Optimization for Linear Regression: A Study Guide

## I. Course Overview & Core Concepts

This section of the course focuses on optimization techniques for Linear Regression (LR), specifically for tabular data. It revisits fundamental machine learning concepts within this context.

### A. Course Sectioning Context

- **Section 1**: Basic Supervised Regression on Tabular Data
- **Section 2**: Basic Deep Learning on 2D Imaging Data
- **Section 3**: Advanced Supervised Learning on Tabular Data
- **Section 4**: Generative and Deep Learning on 1D Sequence Text Data
- **Section 5**: Unsupervised Learning (Mostly on Tabular Data)

### B. Regression Fundamentals

- **Regression**: Deals with predicting a continuous output variable (y).
- **Linear Regression (LR)**: Models the output (Y) as a weighted linear sum of input features (Xs).
- **Sum of Squared Error (Least Squares)**: The standard cost function used to measure the difference between predicted and actual values in LR.
- **Metrics, Implementation, Regression Coefficients (w, b)**: Key aspects to consider in LR.

### C. Machine Learning Task Components

- **Task**: Predicting 'y'.
- **Representation**: Input features 'x' and the function 'f()'.
- **Score Function**: The loss function 'L()'.
- **Search/Optimization**: The process of finding 'argmin()' for the loss function.
- **Models, Parameters**: The chosen model and its adjustable parameters.
- **Data**: 'X', typically tabular.
- **Prediction**: ŷ = f(x) = θ^T x (where θ represents the parameters/weights).

### D. Optimization Principles

- **Objective Function**: The function to be minimized or maximized (e.g., the cost function).
- **Variables**: The parameters being optimized (e.g., weights 'w' or 'θ').
- **Constraints**: Conditions that the variables must satisfy.
- **Goal**: Find variable values that optimize the objective function while satisfying constraints.

## II. Methods for Optimizing Linear Regression

The lecture covers two main approaches to optimizing Linear Regression: Direct Optimization (Normal Equation) and Iterative Optimization (Gradient Descent variants).

### A. Method 1: Directly Optimize (Normal Equation)

**Concept**: For convex functions (like the Sum of Squared Error in LR), the minimum occurs where the derivative (slope) is zero.

**Process**:
1. Write the cost function J(θ) in matrix form.
2. Take the gradient of J(θ) with respect to θ and set it to zero.
3. Solve the resulting equation for θ.

**Formula**: θ = (X^T X)^(-1) X^T y

**Pros**: A single-shot algorithm, easiest to implement.

**Cons**:
- Requires computing the pseudo-inverse (X^T X)^(-1), which can be computationally expensive (especially for large datasets).
- Prone to numerical issues if the matrix is singular or ill-conditioned.

**Direct vs. Iterative**: A direct method, meaning it finds the solution in a finite number of steps.

### B. Method 2: Iteratively Optimize via Gradient Descent

**Concept**: An iterative algorithm that moves towards the minimum of a function by taking steps proportional to the negative of the gradient at the current point.

**Gradient Definition**: A vector whose entries contain the partial derivatives of the function with respect to each variable. It points in the direction of the greatest rate of increase of the function.

**General Algorithm**:
1. Initialize k=0, choose an initial x₀ (randomly or by prior).
2. While k < k_max (or until a stopping criterion is met): x_k = x_{k-1} - α∇_x F(x_{k-1})

**Key Factors Influencing GD**:
- **Learning Rate (α)**: Determines the step size. A critical parameter.
- **Objective Function**: Its shape (e.g., convexity) affects convergence.
- **Starting Point**: Can influence which local minimum is found (though not an issue for LR's convex cost function).
- **Stop Criterion**: When to stop the iterative process.

**Pros**:
- Works on any objective function as long as its gradient can be evaluated.
- Useful for minimizing complex functions.
- Guaranteed convergence for convex functions like LR's cost function.

**Cons**:
- Can be slow converging (especially batch GD).
- Can get stuck in local minima for non-convex functions (not an issue for LR).
- Learning rate selection is crucial.

### C. Variants of Gradient Descent

#### 1. Batch Gradient Descent (GD)

**Concept**: Calculates the gradient of the cost function with respect to the parameters for all training examples in each iteration.

**Update Rule**: θ_{t+1} = θ_t - α∇_θ J(θ_t) = θ_t + αX^T (y - Xθ_t)

**Characteristics**:
- Computes the exact gradient in each step.
- Smooth convergence path.
- Can be computationally expensive for very large datasets, as it needs to process the entire dataset per update.

#### 2. Stochastic Gradient Descent (SGD)

**Concept**: Calculates the gradient and updates the parameters for each individual training example (or a very small subset) at a time.

**Update Rule** (for a single training point i): θ_{t+1} = θ_t + α(y_i - x_i^T θ_t)x_i

**Characteristics**:
- "Noisier" updates due to processing one example at a time.
- Faster per-step computation compared to batch GD.
- Can escape shallow local minima due to the noise (advantage for non-convex problems).
- Useful for massive datasets that don't fit in memory or for online/streaming data.

**Epoch**: One complete pass over the entire training data when using SGD for offline training.

**Pros**:
- Low per-step cost.
- Can be faster converging to an approximate solution.
- Less prone to getting stuck in local optima (for non-convex cases).
- Efficient for large datasets and online learning.

**Cons**:
- Convergence to the exact optimum is not always guaranteed (it tends to oscillate around the minimum).
- More variation in the objective function during training.

#### 3. Mini-Batch Gradient Descent

**Concept**: A compromise between Batch GD and SGD. Computes the gradient on a small "mini-batch" of samples (e.g., B=32, 64, 128).

**Update Rule**: θ_{t+1} = θ_t + (1/B)αX_B^T (y_B - X_B θ_t) (where B refers to the mini-batch size and X_B, y_B are the mini-batch data)

**Characteristics**:
- Combines benefits of both: more stable updates than SGD, but computationally more efficient than Batch GD (especially with GPU acceleration).
- Commonly used in practice.
- Special Cases: B=1 is standard SGD, B=N (total data size) is standard Batch GD.

**Pros**:
- Much faster computationally than single-point SGD (better use of computer architecture like GPUs).
- Low per-step cost.
- Fast convergence.

### D. Stopping Criteria for Gradient Descent

- **Predetermined Maximum Number of Iterations**: A common and simple stopping rule.
- **Threshold for Improvement**: Stop when the improvement in the objective function (e.g., MSE) drops below a certain threshold.
- **Monitoring Training Error**: Plotting Train-MSE per sample to observe its decrease over epochs is helpful for understanding behavior.

## Quiz: Optimization for Linear Regression

1. What is the primary goal of optimization in the context of Linear Regression?
2. Briefly explain the main idea behind the "Normal Equation" method for Linear Regression.
3. What are two significant disadvantages of using the Normal Equation, especially for large datasets?
4. Define "Gradient Descent" as an iterative optimization algorithm.
5. List three critical factors that influence the behavior and performance of Gradient Descent.
6. How does Stochastic Gradient Descent (SGD) differ from Batch Gradient Descent (GD) in terms of gradient calculation and parameter updates?
7. Provide two advantages of using Stochastic Gradient Descent (SGD) compared to Batch Gradient Descent for very large datasets.
8. What is "Mini-Batch Gradient Descent," and why is it often preferred in practice?
9. Explain the concept of an "epoch" in the context of Stochastic Gradient Descent.
10. Name two common stopping criteria for iterative optimization algorithms like Gradient Descent.

## Quiz Answer Key

1. **Primary Goal of Optimization**: The primary goal of optimization in Linear Regression is to find the set of model parameters (weights 'w' or 'θ') that minimize the chosen cost function, typically the Sum of Squared Error (Least Squares). This minimization results in the best-fit line or hyperplane for the given data.

2. **Normal Equation Method**: The Normal Equation method directly solves for the optimal Linear Regression parameters by setting the gradient of the cost function to zero. It analytically calculates the parameters in a single step using matrix operations, without needing an iterative process.

3. **Disadvantages of Normal Equation**: Two significant disadvantages of the Normal Equation are its high computational cost due to the requirement of computing a matrix inverse, particularly for large numbers of features, and its susceptibility to numerical issues if the (X^T X) matrix is singular or ill-conditioned.

4. **Gradient Descent Definition**: Gradient Descent is an iterative optimization algorithm that minimizes an objective function by repeatedly moving in the direction opposite to the gradient of the function. Each step adjusts the parameters by a small amount proportional to the negative of the gradient, gradually approaching a local (or global) minimum.

5. **Critical Factors for GD**: Three critical factors influencing Gradient Descent are the learning rate (α), which determines the step size; the shape of the objective function, which dictates the gradient landscape; and the starting point, which can affect the path to the minimum or which local minimum is found.

6. **SGD vs Batch GD**: Stochastic Gradient Descent (SGD) updates parameters using the gradient calculated from a single training example at a time, leading to noisy but frequent updates. In contrast, Batch Gradient Descent (GD) calculates the gradient using all training examples before making a single parameter update, resulting in smoother but less frequent updates.

7. **Advantages of SGD**: Two advantages of SGD for very large datasets are its low per-step computational cost, as it processes only one sample at a time, making it suitable for data that doesn't fit in memory, and its ability to converge faster to an approximate solution.

8. **Mini-Batch Gradient Descent**: Mini-Batch Gradient Descent is an optimization technique that computes the gradient and updates parameters using a small, randomly selected subset (mini-batch) of the training data. It is often preferred because it balances the stability of Batch GD with the computational efficiency of SGD, leading to faster training and better utilization of hardware like GPUs.

9. **Epoch Concept**: In the context of Stochastic Gradient Descent, an "epoch" refers to one complete pass through the entire training dataset. If SGD is used for offline training, it repeatedly cycles through all samples in the dataset, and each such pass constitutes an epoch.

10. **Stopping Criteria**: Two common stopping criteria for Gradient Descent algorithms include a predetermined maximum number of iterations (epochs) and stopping when the improvement in the objective function (e.g., Train MSE) between consecutive iterations drops below a specified small threshold.