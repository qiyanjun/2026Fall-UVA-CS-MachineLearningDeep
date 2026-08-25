---
layout: post
title: Linear Prediction with Regularization
lecture: S1-lrRegularized
lectureVersion: current
extraContent: 
morenotes: <a href="http://www.stat.cmu.edu/~ryantibs/datamining/lectures/16-modr1.pdf"> More Ridge </a> 
video: <a href="https://youtu.be/w3XwJc8JR1M"> M1</a> + <a href="https://youtu.be/n2c8l0zxvYQ"> M2</a> + <a href="https://youtu.be/BJ83zwSoJzY"> Extra M3</a> 
notes:
categories: tabular
tags:
- 2Regression
- Optimization
- Regularization
- ModelSelection
---

- Notebook Resources: [notebook/L7_regularizedRegression_06_Linear_Regression.ipynb](https://github.com/qiyanjun/2026Fall-UVA-CS-MachineLearningDeep/blob/main/notebook/L7_regularizedRegression_06_Linear_Regression.ipynb)



# Study Guide: Regularized Linear Regression

This guide provides a comprehensive review of the concepts surrounding regularized multivariate linear regression, based on the provided source material. It includes a quiz to test understanding, essay questions for deeper exploration, and a glossary of key terminology.

## Quiz

Answer the following questions in 2-3 sentences each, based on the provided source material.

1. Why is standard linear regression often problematic for datasets where the number of features (p) is greater than the number of samples (n)?
2. What is regularization in the context of machine learning, and what problem does it primarily address?
3. Describe the penalty term used in Ridge (L2) regression and explain how it affects the model's coefficients.
4. Describe the penalty term used in LASSO (L1) regression and explain its key advantage over Ridge regression.
5. What is the primary purpose of the regularization parameter, λ (lambda), and how is its optimal value typically determined?
6. Explain the concept of a "regularization path" and what it visualizes.
7. What is Elastic Net regularization and why was it developed?
8. From a geometric perspective, why does LASSO (L1) regularization tend to produce sparse models (i.e., set some coefficients to exactly zero)?
9. What are the three main goals for a trained machine learning model mentioned in the lecture?
10. What is the "grouping effect" in the context of Elastic Net, and when is it particularly useful?

---

## Quiz Answer Key

1. **When p > n, the matrix X^T X in the normal equation for linear regression has less than full column rank and is therefore not invertible.** This prevents the direct calculation of a unique solution for the regression coefficients using Ordinary Least Squares.

2. **Regularization is an additional criterion added to the loss function to ensure the model does not overfit the training data.** It primarily addresses the issue of overfitting by keeping the model's parameters more "normal" or regular, effectively adding a bias that prefers certain types of weights (e.g., smaller ones) over others.

3. **Ridge regression adds an L2 penalty, which is proportional to the sum of the squared coefficient weights (λ Σ βj²).** This penalty shrinks the coefficients towards zero, with the effect becoming stronger as λ increases. However, it does not set the coefficients exactly to zero.

4. **LASSO regression uses an L1 penalty, proportional to the sum of the absolute values of the weights (λ Σ |βj|).** Its key advantage is that it performs both shrinkage and variable selection simultaneously, as the penalty can force some coefficients to be exactly zero, creating a more interpretable and sparse model.

5. **The regularization parameter λ controls the strength of the penalty; λ=0 corresponds to standard least squares, while a large λ pushes coefficients towards zero.** The optimal value is chosen to ensure good generalization ability on new data and is typically determined using k-fold cross-validation.

6. **A regularization path is a plot that visualizes how the model's regression coefficients (βj) change in value as the regularization parameter λ is varied.** It illustrates the effect of the penalty on each feature's weight, from the unregularized solution to a highly constrained one.

7. **Elastic Net is a hybrid regularization method that combines both L1 (LASSO) and L2 (Ridge) penalties.** It was developed to address disadvantages of LASSO, such as its limitation of selecting at most n variables in a p>n scenario and its tendency to arbitrarily select only one variable from a group of highly correlated ones.

8. **Geometrically, the L1 constraint region is a diamond shape with sharp corners that lie on the axes.** The elliptical contour lines of the least squares error function are more likely to first make contact with this region at one of its corners. This contact point corresponds to a solution where coefficients for other axes are zero, thus producing sparsity.

9. **The three main goals for a trained model are to (1) generalize well to new data, (2) be computationally scalable and efficient, and (3) be trustworthy, meaning it is robust and interpretable.**

10. **The "grouping effect" is a property of Elastic Net that encourages the model to select or deselect groups of highly correlated variables together.** This is useful when features are naturally grouped, as LASSO by itself tends to arbitrarily select only one variable from such a group.

---

## Essay Questions

The following questions are designed for longer, essay-style responses to encourage a deeper synthesis of the material. No answers are provided.

1. **Compare and contrast Ridge, LASSO, and Elastic Net regularization.** Discuss the mathematical formulation of their penalty terms, their impact on model coefficients, and the specific scenarios where one might be preferred over the others.

2. **Explain the problem of overfitting in multivariate linear regression, particularly in the context of "large p, small n" datasets.** How does regularization serve as a solution to this problem, and what is the trade-off involved?

3. **Describe the role and importance of the hyperparameter λ in regularized regression.** Detail the process of using k-fold cross-validation to select an optimal λ and explain why this process is crucial for building a model that generalizes well.

4. **Using the geometric interpretation of regularized regression, explain why L1 regularization (LASSO) leads to sparse solutions while L2 regularization (Ridge) does not.** Use the concepts of constraint regions and objective function contour lines in your explanation.

5. **Discuss the practical application of regularized regression in a real-world problem like text regression for predicting movie revenues.** Explain how model interpretability, a key feature of LASSO and Elastic Net, is valuable in such a domain.

---

## Glossary of Key Terms

| Term | Definition |
|------|------------|
| **Regularization** | An additional criterion added to a model's loss function to prevent overfitting by keeping the parameters "regular" or small. |
| **Overfitting** | A state where a model is too complex and learns "noise" from the training data, resulting in low training error but high error on new, unseen data. |
| **Underfitting** | A state where a model is too simple, resulting in high errors on both the training and test datasets. |
| **Ridge Regression (L2)** | A regularized linear regression method that adds a penalty proportional to the sum of the squared coefficient weights (λ Σ βj²). It performs parameter shrinkage. |
| **LASSO (L1)** | Stands for "Least Absolute Shrinkage and Selection Operator." A regularized regression method that adds a penalty proportional to the sum of the absolute values of the weights (λ Σ \|βj\|). It performs both shrinkage and variable selection. |
| **Elastic Net** | A regularized regression method that combines the L1 and L2 penalties. It was developed to handle highly correlated variables and p>n scenarios more effectively than LASSO alone. |
| **Normal Equation** | The analytical solution for Ordinary Least Squares regression: β = (XᵀX)⁻¹ Xᵀy. It is not solvable if XᵀX is non-invertible. |
| **p > n problem** | A common scenario in datasets, such as in gene expression or text analysis, where the number of features (p) is greater than the number of data samples (n). |
| **λ (Lambda)** | The regularization parameter, also known as a hyperparameter, that controls the strength of the penalty term in regularized models. Its value is often chosen via cross-validation. |
| **K-fold Cross Validation** | A technique used to evaluate a model's ability to generalize to new data and to select optimal hyperparameters like λ. |
| **Regularization Path** | A plot that shows how the values of the estimated coefficients (βj) change as the regularization parameter λ is varied from zero to a large value. |
| **Shrinkage** | The effect of regularization where coefficient values are reduced in magnitude, pulling them closer to zero. This helps to reduce model complexity. |
| **Sparsity** | A property of a model where many of its coefficients are exactly zero. This is a primary feature of LASSO, which effectively performs feature selection. |
| **Grouping Effect** | A property of Elastic Net where it tends to select groups of highly correlated predictor variables together, rather than arbitrarily choosing one from the group as LASSO might. |
| **Pearson Correlation (r)** | A statistical measure of the linear correlation between two variables, yielding a value between -1 (total negative correlation) and +1 (total positive correlation). |