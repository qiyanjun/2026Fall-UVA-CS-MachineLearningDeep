---
LOrder: 230
layout: post
title: Feature Selection 
lecture: S1-feaSelc
extra: true
lectureVersion: current
video: (<b>Extra M2</b> <a href="https://youtu.be/rL82C_sH0xI" target="_blank">Original</a> · <a href="https://youtu.be/AnQPAPqwgi0" target="_blank">AI-Clone</a> · <a href="https://youtu.be/pMtlb-JoR-c" target="_blank">Author-Voice</a>  |  <b>Extra M3</b> <a href="https://youtu.be/aT8Q6DTW1rM" target="_blank">Original</a> · <a href="https://youtu.be/0TaeJZt1Km0" target="_blank">AI-Clone</a> · <a href="https://youtu.be/siYkOA_yv9Q" target="_blank">Author-Voice</a>)
notes: Isabelle Guyon & André Elisseeff (2003), <a href="https://www.jmlr.org/papers/volume3/guyon03a/guyon03a.pdf" target="_blank">An Introduction to Variable and Feature Selection</a> · Hastie, Tibshirani & Friedman, <a href="https://web.stanford.edu/~hastie/ElemStatLearn/" target="_blank">The Elements of Statistical Learning, Ch. 3.3-3.4</a>
morenotes: S3-QuizReview +  <a href="https://web.stanford.edu/~hastie/ElemStatLearn/">ELS Ch3.4 and Ch3.3</a> + <a href="https://scikit-learn.org/stable/modules/feature_selection.html#feature-selection-as-part-of-a-pipeline"> API </a>
categories: tabular
tags:
- ModelSelection
---



# Study Guide: Feature Selection in Machine Learning

---

## The Purpose and Benefits of Feature Selection

In machine learning, datasets can contain thousands or even millions of low-level features. Feature selection is the process of selecting a subset of p' original features from a total of p features. The primary goal is to identify the most relevant features to build better learning models for tasks like classification and regression.

This process yields several significant benefits:

1. **Improved Generalization**: Models with fewer, more relevant features are less sensitive to noise and tend to have lower variance. This aligns with Occam's Razor (the law of parsimony).
2. **Computational Efficiency**: Fewer features reduce training and inference cost and often require fewer labeled examples.
3. **Enhanced Interpretability**: Simpler models are easier to understand and explain, important for trust and transparency.

---

## Core Methodologies for Feature Selection

There are three primary approaches to feature selection, each differing in how they interact with the learning algorithm.

| Approach | Description | Key Characteristics |
|---------|-------------|---------------------|
| **Filter** | Ranks features or feature subsets independently of the predictor. | Learner-agnostic; fast; used as pre-processing. |
| **Wrapper** | Uses a predictor to assess the quality of features or feature subsets. | Learner-dependent; computationally expensive; simple to apply. |
| **Embedded** | Performs feature selection during model training. | Learner-specific; selection integrated into training. |

---

## 1. The Filter Approach

Filter methods select features as a pre-processing step, independently of the classifier. They include univariate and multivariate techniques.

### Univariate Filtering
- **Pearson Correlation**: Measures linear correlation between a feature and the target (range [-1, 1]); detects only linear dependencies.
- **T-test**: Tests whether a feature distinguishes two classes (assumes normality and equal variance). Null hypothesis H0: class means are equal; T-statistic measures significance of mean difference.

### Multivariate Filtering
Methods consider multiple variables jointly to capture interactions missed by univariate methods.

Core challenges:
1. **Scoring Function**: Measure the quality of a subset.
2. **Search Strategy**: Explore the 2^p possible subsets efficiently.

Finding the minimal optimal subset is NP-hard; good heuristics are essential.

---

## 2. The Wrapper Approach

Uses the predictive performance of a specific learner (treated as a black box) to score feature subsets.

Two key questions:
- **(a) Assessment**: How to measure performance?
  - Split into train/validation/test. Train on train, score on validation, choose best subset; optionally use cross-validation; assess final model once on test.
- **(b) Search**: How to explore 2^p subsets?
  - Heuristics:
    - **Forward Selection**: Start empty; add features iteratively.
    - **Backward Elimination**: Start full; remove features iteratively.
    - **Advanced**: Beam search, GSFS, PTA(l,r), floating search.

Main criticism: high computational cost due to repeated training/evaluation.

---

## 3. The Embedded Approach

Integrates selection into training a single model; selection occurs implicitly.

- **Key Characteristics**: Learner-specific.
- **Example**: **L1-regularization** shrinks many coefficients to zero, selecting features.

---

## Feature Selection vs. Feature Extraction

- **Feature Selection**: Selects a subset of original features  
  (X1, ..., Xp → Xk1, ..., Xkp'); features remain unchanged.
- **Feature Extraction / Dimensionality Reduction**: Creates new features as (possibly weighted) combinations of originals  
  (X1, ..., Xm → g1(X), ..., gp'(X)).

Example: **Principal Component Analysis (PCA)** finds a linear mapping to a lower-dimensional space that maximally preserves variance; principal components are combinations of original features.

---

## Model Selection and Assessment

Feature selection is tied to broader tasks of model selection and assessment.

- **Model Selection**: Estimate performance to choose the best model (e.g., hyperparameters or feature subset).
- **Model Assessment**: Estimate the chosen model’s prediction error on unseen data.

### Data Handling Strategies
- **Data-Rich Scenario**: Split into train (build), validation (select), test (final assess).
- **Insufficient Data**: Use cross-validation or bootstrap; or analytical approximations (AIC/BIC) to mimic validation.

---

## Review Quiz

Answer the following questions in 2-3 sentences each based on the provided material.

1. What is the primary objective of feature selection in machine learning?
2. According to the text, what are the three main benefits of applying feature selection?
3. Explain the fundamental difference between the Filter and Wrapper approaches.
4. What is a major limitation of univariate filtering methods like the Pearson Correlation?
5. Why is the task of finding a minimal optimal feature subset described as NP-hard?
6. In the Wrapper approach, what are the distinct roles of the training, validation, and test data sets?
7. Describe the core ideas behind the Forward Selection and Backward Elimination search strategies.
8. How does the Embedded approach to feature selection differ from both Filter and Wrapper methods?
9. Distinguish between Feature Selection and Feature Extraction, using Principal Components Analysis (PCA) as an example.
10. What is the purpose of model selection, and how does it differ from model assessment?

---

## Answer Key

1. **Objective**: Select a subset of the most relevant original features to build models that are better, faster, and more interpretable for classification or regression.
2. **Benefits**: Improved generalization (lower variance, less noise sensitivity); computational efficiency (faster training/inference); enhanced interpretability (more explainable).
3. **Filter vs Wrapper**: Filter ranks/selects features independently of the learner as pre-processing; Wrapper uses the performance of a specific learner to score feature subsets.
4. **Limitation of Univariate Filtering**: Detects only linear, single-feature relationships; may miss features useful only in combination.
5. **NP-hardness**: With p features, there are 2^p subsets; exhaustive search is infeasible, requiring heuristics.
6. **Data splits**: Train to fit predictors per subset; Validation to select the best subset; Test used once for final unbiased performance estimate.
7. **Search strategies**: Forward Selection adds features iteratively from empty; Backward Elimination removes features iteratively from full.
8. **Embedded**: Selection is integrated into model training (learner-specific), unlike separate pre-processing (Filter) or external evaluation loops (Wrapper).
9. **Selection vs Extraction**: Selection keeps original features; Extraction creates new features (e.g., PCA components) as combinations that preserve variance.
10. **Model Selection vs Assessment**: Selection chooses the best model via validation; Assessment estimates final generalization on unseen data (test set).

---

## Essay Questions

Formulate comprehensive responses for each:

1. Discuss Occam's Razor and its relationship to the three primary goals of feature selection.
2. Compare univariate vs. multivariate Filter methods; give a scenario where multivariate succeeds but univariate fails.
3. With millions of features and limited compute, which approach (Filter, Wrapper, Embedded) would you choose and why? Discuss trade-offs.
4. Explain the inherent “search problem” in feature selection. Describe three heuristic strategies and when each is preferable.
5. Detail the complete Wrapper process with emphasis on data splitting. Why separate validation and test sets to avoid bias?

---

## Glossary of Key Terms

| Term | Definition |
|------|------------|
| **Backward Elimination** | Heuristic that starts with all features and iteratively removes the least useful ones. |
| **Cross-Validation** | Efficient sample reuse to estimate learner performance when validation data is limited. |
| **Dimensionality Reduction** | Creating new features g(X) by combining originals; also called Feature Extraction. |
| **Embedded Approach** | Selection performed implicitly during model training; learner-specific. |
| **Feature Extraction** | Transforming X into new features g(X), often lower dimensional. |
| **Feature Selection** | Selecting a subset of p' original features from p available features. |
| **Filter Approach** | Ranks features/subsets independent of the predictor as pre-processing. |
| **Forward Selection** | Heuristic that starts empty and iteratively adds the most useful features. |
| **L1-Regularization** | Embedded method that penalizes complexity and drives some coefficients to zero. |
| **Model Assessment** | Estimating prediction error of the final model on new data. |
| **Model Selection** | Estimating performance of candidate models to choose the best. |
| **Multivariate Methods** | Filter methods assessing multiple variables jointly. |
| **NP-hard** | Class of problems lacking known polynomial-time solutions; subset search is NP-hard. |
| **Occam's Razor** | Prefer simpler explanations/models with fewer assumptions. |
| **Pearson Correlation** | Measures linear relationship between two variables (range [-1, 1]). |
| **Principal Components Analysis (PCA)** | Feature extraction mapping to a lower-dimensional space preserving variance. |
| **T-test** | Univariate test assessing whether a feature separates two normally distributed classes. |
| **Univariate Methods** | Filter methods assessing one variable at a time. |
| **Wrapper Approach** | Uses a learner as a black box to score feature subsets by predictive power. |