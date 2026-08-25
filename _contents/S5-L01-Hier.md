---
layout: post
title: Clustering Hier
lecture: S5-clustering1-Hier
lectureVersion: current
video: <a href="https://youtu.be/mxrcPHWCZAI">M1</a> + <a href="https://youtu.be/2K6gXLf4Em4">M2</a>  
notes: <a href="https://scikit-learn.org/stable/auto_examples/cluster/plot_linkage_comparison.html#sphx-glr-auto-examples-cluster-plot-linkage-comparison-py"> compare Hier clusterings</a>
categories: tabular
tags:
- 4Unsupervised
---

# Summary: 

Hierarchical: 
- Module1: Basics of Unsupervised Clustering @ https://youtu.be/mxrcPHWCZAI
- Module2: Hierarchical Clustering @ https://youtu.be/2K6gXLf4Em4

K-means
- Module1: Basic K-Means Algorithm @ https://youtu.be/239r6zlZYhY 
- Module2: Theory/Complexity of K-Means @ https://youtu.be/OFKKeVhCQfA
- Extra Module3: Gaussian Mixture Models @ https://youtu.be/pmNPXpj0eK4



# Unsupervised Learning: A Study Guide on Hierarchical Clustering

## Quiz: Short Answer Questions

Answer the following questions in 2-3 sentences based on the provided course material.

1. What is the core difference between unsupervised and supervised machine learning?
2. Define the primary goal of clustering and describe the two main properties of an ideal clustering solution.
3. List the four desirable properties of a distance measure and provide a brief, intuitive explanation for one of them.
4. What is the Minkowski metric, and what are its three most common forms?
5. Under what conditions is Hamming distance used, and how does it relate to the Manhattan distance?
6. Explain the fundamental difference between the operational approaches of partitional and hierarchical clustering algorithms.
7. What is a dendrogram, and what role does it play in hierarchical clustering?
8. Describe the process of bottom-up agglomerative clustering.
9. Name and briefly define the three common methods for calculating the distance between two clusters in hierarchical clustering.
10. What is the primary limitation of hierarchical clustering methods regarding their performance on large datasets?

---

## Answer Key

1. Unsupervised learning operates on raw, unlabeled data (an unlabeled dataset X with no Y variable), aiming to find inherent structures or patterns. In contrast, supervised learning is provided with labeled examples (both X and Y variables) and aims to learn a function that maps inputs to outputs, such as in regression (continuous Y) or classification (discrete Y).

2. The goal of clustering is to group a set of objects into classes where objects within a class are similar to each other and dissimilar to objects in other classes. An ideal solution achieves high intra-class similarity (minimized intra-cluster distances) and low inter-class similarity (maximized inter-cluster distances).

3. The four properties are: Symmetry, Constancy of Self-Similarity, Positivity Separation, and Triangular Inequality. Symmetry (D(A,B) = D(B,A)) provides the intuition that if "Alex looks like Bob," then "Bob must look like Alex."

4. The Minkowski metric is a generalized formula for calculating the distance between two objects, x and y, that each have p features. Its three most common forms are determined by the parameter 'r': Euclidean distance (r=2), Manhattan distance (r=1), and "sup" distance (r=+∞).

5. Hamming distance is used when all features of the data are binary or discrete, such as gene expression levels recorded as either high (1) or low (0). It is a special case of the Manhattan distance applied to this type of data.

6. Partitional algorithms typically start with a random partitioning of the data into a pre-specified number of clusters and then refine these clusters iteratively. Hierarchical algorithms do not require a fixed number of clusters in advance and instead build a tree-based structure, either by starting with individual objects and merging them (agglomerative) or by starting with a single group and splitting it (divisive).

7. A dendrogram is a tree-based diagram that represents the hierarchical taxonomy produced by a clustering algorithm. It illustrates the history of merges or splits, showing which objects and clusters were grouped together and at what distance, allowing users to form partitions by "cutting" the tree at a desired level.

8. Bottom-up agglomerative clustering begins with each data object as its own separate cluster. In each step, the algorithm identifies the two closest clusters based on a chosen distance metric and merges them into a single new cluster, repeating this process until only one cluster containing all objects remains.

9. The three methods are Single-Link, Complete-Link, and Average-Link. Single-Link defines cluster distance as the distance between the two closest members of the clusters. Complete-Link uses the distance between the two farthest members. Average-Link calculates the average distance of all cross-cluster pairs.

10. The primary limitation is that they do not scale well to large datasets. The time complexity is at least O(n²), where n is the number of objects, because in the first iteration, the algorithm must compute the similarity for all pairs of instances.

---

## Essay Questions

The following questions are designed for a more in-depth, essay-format response. No answers are provided.

1. Discuss the subjective nature of clustering. Using the concepts of "groupness," "similarity/distance," and the determination of the number of clusters, explain why different yet equally valid clustering solutions can exist for the same dataset.

2. Compare and contrast the use of Euclidean distance and the Pearson correlation coefficient as similarity measures. Explain a scenario where one would be significantly more appropriate than the other, referencing the provided discussion on gene expression time series data.

3. Provide a comprehensive explanation of the bottom-up agglomerative hierarchical clustering process, from the initial pairwise distance matrix to the final dendrogram. In your explanation, detail how the choice between Single-Link, Complete-Link, and Average-Link methods influences the process and the potential shapes of the resulting clusters.

4. Analyze the computational complexity and scalability challenges of hierarchical clustering. Elaborate on why the time complexity is at least O(n²) and can become O(n³) if implemented naively. Discuss the practical implications of this complexity when applying these methods to modern, large-scale datasets.

5. Machine learning is framed as a combination of Representation, a Score Function, and a Search/Optimization strategy. Deconstruct hierarchical clustering using this framework. What constitutes the data representation, what is the search strategy, and why is its score function described as having "no clearly defined loss"?

---

## Glossary of Key Terms

| Term | Definition |
|------|------------|
| **Agglomerative Clustering** | A bottom-up hierarchical clustering method that starts with each object in its own cluster and repeatedly joins the closest pair of clusters until only one remains. |
| **Average-Link** | A method for computing the distance between two clusters based on the average distance of all cross-cluster pairs. It is noted as the most widely used measure and is robust against noise. |
| **Clustering** | The process of grouping a set of objects into classes of similar objects. It is the most common form of unsupervised learning. |
| **Complete-Link** | A method for computing the distance between two clusters based on the distance of their two furthest members. This method tends to produce tight clusters. |
| **Correlation Coefficient** | A similarity measure that quantifies the linear correlation between two sequences, giving a value between +1 (total positive correlation) and -1 (total negative correlation). It is unit-independent. |
| **Data Matrix** | A representation of data in a tabular format with n observations (rows) on p variables (columns). Rows can be called points, instances, or samples, while columns are features, attributes, or predictors. |
| **Dendrogram** | A tree-based hierarchical taxonomy that is the output of a hierarchical clustering algorithm. It visually represents the history of merges or splits and the distances at which they occurred. |
| **Divisive Clustering** | A top-down hierarchical clustering method that starts with all data in a single cluster, considers every possible way to divide it into two, chooses the best division, and recursively operates on the new clusters. |
| **Distance Measure Properties** | A set of four desirable properties for any distance measure: Symmetry (D(A,B) = D(B,A)), Constancy of Self-Similarity (D(A,A) = 0), Positivity Separation (D(A,B) = 0 if and only if A=B), and Triangular Inequality (D(A,B) <= D(A,C) + D(B,C)). |
| **Edit Distance** | A generic technique for measuring similarity by calculating the "effort" it takes to transform one object into another. The measure of effort becomes the distance. |
| **Euclidean Distance** | A common distance measure, corresponding to the Minkowski metric with r=2. It calculates the straight-line distance between two points in a p-dimensional space. |
| **Hamming Distance** | A distance measure used when all features are binary or discrete. It is a special case of the Manhattan distance. |
| **Hierarchical Clustering** | An algorithm that builds a tree-based hierarchy (dendrogram) from a set of objects. It can be performed bottom-up (agglomerative) or top-down (divisive). |
| **Inter-cluster Distance** | The distance between different groups or clusters. An effective clustering algorithm seeks to maximize this distance. |
| **Intra-cluster Distance** | The distance between data points within the same group or cluster. An effective clustering algorithm seeks to minimize this distance. |
| **Manhattan Distance** | A common distance measure, corresponding to the Minkowski metric with r=1. It is calculated as the sum of the absolute differences between the coordinates of two points. |
| **Minkowski Metric** | A generalized formula for defining the distance between two objects in a p-dimensional space, controlled by a parameter 'r'. |
| **Partitional Algorithms** | A class of clustering algorithms (e.g., K-means) that usually start with a random partitioning of the data and refine it iteratively to find the clusters. |
| **Single-Link** | A method for computing the distance between two clusters based on the distance of their two closest members (nearest neighbors). This method can result in long and skinny clusters. |
| **Unsupervised Learning** | A type of machine learning that involves learning from raw, unlabeled, or unannotated data, as opposed to supervised learning where the label of each example is given. |
