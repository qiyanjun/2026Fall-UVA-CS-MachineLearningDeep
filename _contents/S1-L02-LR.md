---
LOrder: 60
layout: post
title: Linear Regression
lecture: S1-LinearReg
lectureVersion: next
extraContent:
notes: 'Hastie, Tibshirani & Friedman (2009), <a href="https://web.stanford.edu/~hastie/ElemStatLearn/" target="_blank">The Elements of Statistical Learning, Ch. 3: Linear Methods for Regression</a> · Powell & Lehe, <a href="https://setosa.io/ev/ordinary-least-squares-regression/" target="_blank">Ordinary Least Squares Regression, Explained Visually</a>'
morenotes: <a href="https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html">linear regression coderun</a>
video: <b>M1</b> <a href="https://youtu.be/0n2WzBzgzOE" target="_blank">Original</a> · <a href="https://youtu.be/3LiGR1SK09E" target="_blank">AI-Clone</a>  |  <b>M2</b> <a href="https://youtu.be/dArbWu7pfYc" target="_blank">Original</a> · <a href="https://youtu.be/izOijEDXVyg" target="_blank">AI-Clone</a>  |  <b>M3</b> <a href="https://youtu.be/m1xIIvLu3Tw" target="_blank">Original</a> · <a href="https://youtu.be/FxeQwF11a0c" target="_blank">AI-Clone</a>  |  <b>M4</b> <a href="https://youtu.be/4wXFrsM-Zbw" target="_blank">Original</a> · <a href="https://youtu.be/9qm18HrRPQg" target="_blank">AI-Clone</a>
categories: tabular
tags:
- 2Regression
- Linear
- 1Basic
notebooks: '<a href="/2026Fall-UVA-CS-MachineLearningDeep//notebook/L3_plot_ols.ipynb" target="_blank">notebook/L3_plot_ols.ipynb</a>'
---


# Understanding Linear Regression: A Study Guide

## Quiz

**Instructions**: Answer the following questions in 2-3 sentences each.

1. What is the primary goal of Machine Learning in a nutshell, as described in the lecture?
2. Define "regression" in the context of machine learning, specifically mentioning the nature of the target variable.
3. Explain the purpose of the "score function" (or loss/cost function) in linear regression.
4. Describe what the training dataset consists of for supervised regression.
5. In the linear regression equation f(x) = wx + b, what do w and b represent?
6. How does the concise notation ŷ = f(x) = x^T θ = θ^T x simplify the representation of linear regression with multiple features?
7. What is the "Sum of Squared Error" (SSE) and why is it used as a loss function in linear regression?
8. Briefly explain the concept of "Normal Equations" in the context of finding optimal regression coefficients.
9. Why is it important for a learned model to "Generalize Well" to unseen data?
10. What are the three main aims for a learned model besides generalizing well, as discussed in the lecture?

## Answer Key

1. **Primary Goal of Machine Learning**: The primary goal of Machine Learning is to optimize a performance criterion using example data or past experience. The ultimate aim is to generalize this learned knowledge effectively to unseen data.

2. **Regression Definition**: In machine learning, regression refers to tasks where the target variable (Y) is continuous. This means the model predicts a numerical value rather than a category or class.

3. **Score Function Purpose**: The score function (or loss/cost function), denoted as L(), quantifies the difference between the model's predictions and the actual target values. The goal of the learning algorithm is to minimize this function, thereby finding the optimal model parameters.

4. **Training Dataset**: The training dataset for supervised regression consists of input-output pairs. These pairs include available examples (input features, X) and their corresponding continuous target labels (output, Y), which the model uses to learn.

5. **Linear Regression Parameters**: In the linear regression equation f(x) = wx + b (for a single variable), w represents the weight or slope, indicating how much Y changes for a unit change in X. b represents the bias or intercept, which is the value of Y when X is zero.

6. **Concise Notation**: The concise notation ŷ = f(x) = x^T θ = θ^T x uses vector/matrix products to represent multiple features and their corresponding weights in a single expression. This compact form simplifies mathematical derivations and computational implementation for multivariate linear regression.

7. **Sum of Squared Error (SSE)**: The Sum of Squared Error (SSE) measures the sum of the squared differences between the predicted values (f(x)) and the actual target values (y). Squaring the errors ensures that all errors contribute positively to the total loss and penalizes larger errors more heavily, providing a clear objective for minimization.

8. **Normal Equations**: The Normal Equations provide a closed-form solution to find the optimal regression coefficients (θ) by setting the derivative (gradient) of the cost function with respect to θ to zero. This method directly calculates the θ that minimizes the SSE for linear regression.

9. **Generalization**: Generalizing well means that the model performs accurately not only on the data it was trained on but also on new, unseen data. This indicates that the model has learned underlying patterns rather than just memorizing the training examples, making it useful in real-world applications.

10. **Additional Model Aims**: Besides generalizing well, the lecture suggests that a learned model should also be computationally scalable and efficient, and robust, trustworthy, and interpretable. These attributes contribute to the practical utility and reliability of machine learning models.


# Linear Regression Basics

## 1. Machine Learning in a Nutshell

**Definition**: Machine Learning (ML) is a field that grew out of Artificial Intelligence (AI). Its core purpose is to "Optimize a performance criterion using example data or past experience, aiming to generalize to unseen data."

**Key Components of ML**: The process of machine learning can be broken down into several fundamental elements:

- **Task**: What needs to be predicted or achieved (e.g., y in regression).
- **Representation**: How the data and model are structured (e.g., x, f()).
- **Score Function**: A measure of how well the model is performing (e.g., L()).
- **Search/Optimization**: The method used to find the best model parameters (e.g., argmin()).
- **Models, Parameters**: The specific mathematical structure and adjustable values of the model (e.g., w, b for regression coefficients).
- **Data**: The input used for training and testing (e.g., X: Tabular).

## 2. Regression Fundamentals

**Definition**: Regression specifically deals with predicting a continuous target variable y.

**Linear Regression**: In linear regression, the target Y is modeled as a "Weighted linear sum of Xs."

- For a single variable x, the function is f(x) = wx + b. Here, w is the weight (slope) and b is the bias (intercept). A slope of 2 means "every 1-unit change in X yields a 2-unit change in Y."
- For multiple features x₁, x₂, ..., the model becomes ŷ = f(x) = θ₀ + θ₁x₁ + θ₂x₂. Here, θ₀ is the intercept, and θ₁, θ₂ are coefficients for x₁, x₂ respectively.

**Supervised Learning**: Linear Regression is a type of supervised learning. This means the training dataset consists of "input-output pairs" (X, Y), where Y (the target) is known for each input X.

## 3. Data Representation for Regression

### Tabular Dataset Structure
- **Data/points/instances/examples/samples/records**: Rows of the dataset.
- **Features/attributes/dimensions/independent variables/covariates/predictors/regressors**: Columns of the dataset, excluding the target.
- **Target/outcome/response/label/dependent variable**: A special column (usually the last) that is continuous-valued and to be predicted.

### Concise Vector/Matrix Notation
To simplify computations, especially with multiple features:

- Each data sample x is represented as a column vector.
- A pseudo feature x₀=1 (the intercept term) is added to the feature vector, making it x^T = [(x₀=1), x₁, x₂, …, xₚ].
- The parameter vector θ is also a column vector [θ₀, θ₁, ..., θₚ]^T.
- The linear regression function can then be concisely written as ŷ = f(x) = x^T θ = θ^T x.

**Training Set in Matrix Form**: The entire training set with n samples can be represented as a matrix X (where each row is a feature vector x_i^T) and a target vector y_train = [y₁, y₂, ..., yₙ]^T.

## 4. Training and Optimization

**Training Goal**: To find the optimal parameters (w, b or θ) by minimizing a Loss/Cost function L(). This function quantifies the "difference between y and f(x) on available examples in the training set."

### Sum of Squared Error (SSE) / Mean Squared Error (MSE)
The most common loss function for linear regression:

- J(θ) = 1/2 × Σ (f(xᵢ) - yᵢ)²
- In matrix form, J(θ) = 1/2 × (Xθ - y)^T (Xθ - y)

### Finding Optimal Parameters (θ)

#### Closed-form Solution (Normal Equation)
Since J(θ) is a convex function, its minimum can be found by taking the derivative (gradient) with respect to θ and setting it to zero (∇θJ(θ) = 0). This leads to the "normal equations": X^T Xθ = X^T y.

The solution for θ is then θ* = (X^T X)^(-1) X^T y.

#### Convexity of J()
The loss function J(θ) is convex. "Intuitively, a convex function (1D case) has a single point at which the derivative goes to zero, and this point is a minimum." For multivariate functions, if the Hessian matrix is positive definite (which X^T X is, given certain conditions), the function is convex and the point where the gradient is zero is a minimum.

#### Gradient Descent (GD) / Stochastic Gradient Descent (SGD)
These are iterative optimization algorithms used when a closed-form solution is not feasible or computationally too expensive. (Mentioned in outline, but not detailed in excerpts).

## 5. Evaluating Model Performance

**Generalization**: A primary aim is for the "learned model [to] Generalize Well" to unseen data. It should also be "Computational Scalable and Efficient" and "Robust / Trustworthy / Interpretable."

**Testing Mode**: After training, the model is tested on "Production Data" (also called test data), which consists of new inputs X? for which the output f(X?) is predicted.

### Metrics for Regression
- **Test MSE Error**: A common metric reported for evaluating regression models, calculated on the test set: J_test = 1/m × Σ (x_i^T θ* - y_i)².
- Other metrics exist (referencing scikit-learn documentation).

## 6. Computational Considerations

**Why Vector/Matrix Form?** It allows for "Concise Notation," "Closed form solution" for training, and efficient computation for testing (ŷ_test = X_test θ*).

### Computational Cost (Naive Matrix Multiply)
A naive matrix multiplication C = C + A×B has a complexity of O(n³) flops.

### Optimizing Computational Cost

#### Data Parallelization
Utilizing CPU SIMD (Single Instruction, Multiple Data), multithreading, and GPU parallelization. SIMD allows "one operation [to produce] multiple results."

#### Memory Hierarchy/Locality
Exploiting spatial (accessing nearby data) and temporal (reusing data) locality to improve average memory access speed through caches.

#### Better Algorithms
Examples include Strassen's Matrix Multiply, which reduces complexity to O(n^2.81) (asymptotically faster than naive O(n³)).

#### BLAS (Basic Linear Algebra Subroutines)
Industry-standard interfaces (e.g., numpy wraps BLAS) providing optimized implementations for vector (BLAS1), matrix-vector (BLAS2), and matrix-matrix (BLAS3) operations. BLAS3 is generally preferred for its potential for higher computational intensity.

## 7. Advanced/Optional Concepts

**Probabilistic Interpretation**: Linear regression can be viewed probabilistically by assuming the error term ε (representing unmodeled effects or random noise) follows a Gaussian distribution N(0, σ). This leads to the concept of Maximum Likelihood Estimation for finding θ.

**Gram Matrix**: X^T X is referred to as a Gram Matrix and is always positive semi-definite. If X has full rank, X^T X is positive definite and invertible, ensuring a unique minimum for J(θ).

**Regularization**: Used when X has less than full column rank to handle non-invertible X^T X (not detailed in these excerpts but mentioned as a solution).

**Scalability to Big Data**: While polynomial time algorithms are good, O(n) can still be too slow for truly massive datasets, leading to research in "low rank, sparse, hardware, sampling, randomized…" solutions.
