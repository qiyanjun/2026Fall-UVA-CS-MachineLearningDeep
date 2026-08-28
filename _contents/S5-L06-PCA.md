---
LOrder: 220
layout: post
title: PCA, Feature Selection
lecture: S2-dimReduce
extraContent:  S2-PCA
notes: <a href="https://www.youtube.com/watch?v=FgakZw6K1QQ&list=PLblh5JKOoLUICTaGLRoHQDuF_7q2GfuJF&index=24">Great PCA Video</a> + <a href="https://github.com/qiyanjun/2026Fall-UVA-CS-MachineLearningDeep/blob/main/notebook/L14_PCA_face_IRIS.ipynb"> PCA Notebook </a>
morenotes: <a href="https://umap-learn.readthedocs.io/en/latest/"> UMAP </a>    
video:  <a href="https://youtu.be/-Fk4b1PPoEA"> M1 </a> 
categories: 2D (Vision)
lectureVersion: current
extra: true
tags:
- DimensionReduction
---


- Notebook to try: 

<a href="https://github.com/qiyanjun/2026Fall-UVA-CS-MachineLearningDeep/blob/main/notebook/L14_PCA_face_IRIS.ipynb"> PCA Notebook </a>


# Study Guide: Dimensionality Reduction and PCA


---


## running notebook 

<a href="https://github.com/qiyanjun/2026Fall-UVA-CS-MachineLearningDeep/blob/main/notebook/L14_PCA_face_IRIS.ipynb"> PCA Notebook </a>



## Short-Answer Quiz

**Instructions:** Answer the following questions in 2-3 sentences based on the provided source material.

1. What is the "Curse of Dimensionality," and what is its practical effect on machine learning models?
2. Define the two primary methods of dimensionality reduction: feature extraction and feature selection.
3. Describe the three main approaches to feature selection discussed in the text.
4. What is the main objective of Principal Component Analysis (PCA), and what property of the data does it seek to maximize in its projections?
5. What are the two core components of an autoencoder, and what is its primary function in the context of feature learning?
6. When using PCA to project data onto a single line, what is the formal algebraic goal?
7. How are eigenvalues related to their corresponding principal components (eigenvectors) in terms of explaining the data's structure?
8. List two criteria used to determine how many principal components should be retained after performing PCA.
9. Explain a scenario where the direction of maximum variance found by PCA might not be optimal for a classification task.
10. Briefly outline the process of using the "Eigenfaces" technique for face recognition.

---

## Answer Key

1. **The "Curse of Dimensionality" refers to the issue where the number of training examples required for a model increases exponentially with the number of features (dimensionality).** In practice, including more features does not always improve accuracy and can lead to worse performance.

2. **Feature extraction finds a new set of features by applying a mapping function to the existing features.** Feature selection, in contrast, chooses a subset of the original, existing features without creating new ones.

3. **The three approaches are Filtering, Wrapper, and Embedded.** The Filtering approach ranks features independently of the predictor; the Wrapper approach uses a predictor to assess feature subsets; and the Embedded approach uses a predictor that internally selects a subset of features as it builds a single model.

4. **The objective of PCA is to approximate a high-dimensional dataset with a lower-dimensional linear subspace.** It achieves this by finding new axes (principal components) in the directions of the greatest variability, thereby maximizing the variance of the projected data.

5. **An autoencoder consists of an encoder, which compresses the input into a lower-dimension latent space, and a decoder, which reconstructs the input from that latent space.** Its function is to train the hidden layer units to become good and reliable feature detectors by minimizing the reconstruction loss between the original input and the reconstructed output.

6. **The formal algebraic goal is to find a line (direction vector v) that maximizes the sum of squares of the data samples' projections onto that line.** This is equivalent to finding the eigenvector v associated with the largest eigenvalue λ of the matrix XᵀX.

7. **The principal components (eigenvectors) are the new, uncorrelated features derived by PCA.** The variance of the data along each principal component vₖ is equal to its corresponding eigenvalue λₖ, meaning that PCs with small eigenvalues correspond to directions of small variance in the data.

8. **One method is to keep enough PCs to explain a cumulative variance greater than a certain threshold, such as 50-70%.** A second method is to use a "Scree plot," which visualizes the variance explained by each PC, and keep only the components with eigenvalues above a certain value (e.g., >1).

9. **The direction of maximum variance is not always good for classification.** For example, with two distinct classes represented as ellipsoidal Gaussian densities, the first principal component may capture the overall data trend, while the second, less significant component may be the one that best discriminates between the two classes.

10. **The Eigenfaces technique first computes the principal components ("eigenfaces") from a set of training face images.** All training images are then projected into this new, lower-dimensional "face space." A novel image is classified by projecting it into the same space and finding the nearest neighboring training face based on the distance between their low-dimensional coefficients.

---

## Essay Questions

**Instructions:** The following questions are designed for a more in-depth, essay-style response. Answers are not provided.

1. **Compare and contrast the three primary feature selection strategies:** Filtering, Wrapper, and Embedded. Discuss the relative advantages and disadvantages of each in terms of computational cost, model dependency, and optimization for a specific learning algorithm.

2. **Elaborate on the "Curse of Dimensionality" using the specific examples cited in the text** (QSAR drug screening, Leukemia Diagnosis, and Text Categorization). Explain how high-dimensional feature spaces in these domains necessitate dimensionality reduction and what benefits are gained from applying these techniques.

3. **Provide a detailed explanation of the mathematical and algebraic interpretation of Principal Component Analysis.** Describe the roles of the centered data matrix, the covariance matrix, eigenvalues, and eigenvectors in the process of identifying the principal components that capture maximum variance.

4. **Using the "Eigenfaces" method as a case study, describe the end-to-end application of PCA for image recognition.** Explain how a high-dimensional image is transformed into a low-dimensional representation, how this representation is used for classification, and what information is potentially lost or preserved in this process.

5. **Discuss the limitations of PCA.** Explain why its objective of maximizing variance may not align with the objective of maximizing class separability in a supervised learning context, and illustrate with the examples provided in the source material.

---

## Glossary of Key Terms

| Term | Definition |
|------|------------|
| **Autoencoder** | A neural network structure, trained to reproduce its input, consisting of an encoder and a decoder. It forces the 'hidden layer' units to become reliable feature detectors by minimizing reconstruction loss. |
| **Covariance Matrix** | A matrix representing the covariance between pairs of variables in a dataset. In PCA, the eigenvectors of this matrix (or of XᵀX for centered data) are the principal components. |
| **Curse of Dimensionality** | The phenomenon where the number of training examples required for a model increases exponentially with the number of features (dimensionality). In practice, this can lead to worse performance as more features are added. |
| **Decoder** | A component of an autoencoder that reconstructs the input from the compressed latent space representation generated by the encoder. |
| **Dimensionality Reduction** | The process of choosing an optimum set of features of lower dimensionality to create simpler, more interpretable, and better-generalizing models. |
| **Eigenfaces** | A term for the principal components (eigenvectors) computed from a covariance matrix of face images. This technique constructs a low-dimensional linear subspace to explain variation in a set of face images for recognition tasks. |
| **Eigenvalue (λ)** | A scalar associated with an eigenvector. In PCA, the eigenvalue represents the amount of variance in the data explained by its corresponding eigenvector (principal component). |
| **Eigenvector (v)** | A vector whose direction remains unchanged when a linear transformation is applied to it. In PCA, the eigenvectors of the covariance matrix are the principal components, representing the directions of maximum variance. |
| **Embedded Approach** | A feature selection method where a predictor is used to build a single model with a subset of features that are internally selected during the training process. |
| **Encoder** | A component of an autoencoder that compresses an input into a latent-space representation of a usually smaller dimension. |
| **Feature Extraction** | A method of dimensionality reduction that finds a set of new features by applying a mapping function (which can be linear or non-linear) to the existing features. |
| **Feature Selection** | A method of dimensionality reduction that chooses a subset of the original features. |
| **Filtering Approach** | A feature selection method that ranks features or feature subsets as a pre-processing step, independently of the learning algorithm (predictor) being used. |
| **Principal Component Analysis (PCA)** | An unsupervised linear feature extraction method that approximates a high-dimensional dataset with a lower-dimensional linear subspace. It seeks a projection that preserves as much of the information (variance) in the data as possible. |
| **Principal Components (PCs)** | The new, more informative, uncorrelated features found by PCA, which correspond to the eigenvectors of the data's covariance matrix. The first PC explains the most variance, the second PC explains the next most, and so on. |
| **Wrapper Approach** | A feature selection method that uses a predictor (treated as a "black box") to assess and score different subsets of features based on their predictive power. |

