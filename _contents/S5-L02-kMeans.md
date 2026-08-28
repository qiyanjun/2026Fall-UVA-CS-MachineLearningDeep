---
LOrder: 490
layout: post
title: Clustering Partition
lecture: S5-clustering2-kMeans
lectureVersion: current
extraContent: S5-EMextra
video: <a href="https://youtu.be/239r6zlZYhY">M1</a> + <a href="https://youtu.be/OFKKeVhCQfA">M2</a> 
notes: <a href="https://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_comparison.html"> compare clusterings </a> 
categories: tabular
tags:
- 4Unsupervised
- Generative
---


# Summary: 
- Module1: Basic K-Means Algorithm @ https://youtu.be/239r6zlZYhY 
- Module2: Theory/Complexity of K-Means @ https://youtu.be/OFKKeVhCQfA
- Extra Module3: Gaussian Mixture Models @ https://youtu.be/pmNPXpj0eK4


# K-Means and Partitional Clustering Study Guide

## Quiz: Short Answer Questions

Answer the following questions in 2-3 sentences, based on the provided source material.

1. What is the fundamental difference between unsupervised and supervised machine learning?
2. Describe the primary goal of clustering in terms of intra-cluster and inter-cluster distances.
3. How do partitional clustering algorithms, like K-means, generally operate?
4. Outline the five core steps of the K-means algorithm.
5. What role does the centroid play in the K-means algorithm, and how is its position updated?
6. Explain the concept of "elbow finding" or "knee finding" for determining the number of clusters.
7. Why is the initial selection of cluster centers (seeds) a critical consideration in K-means?
8. What is the mathematical objective of the K-means algorithm, and how does it relate to convergence?
9. Briefly describe the Gaussian Mixture Model (GMM) approach to clustering.
10. Define "purity" as an external criterion for evaluating cluster quality.

---

## Answer Key

1. Unsupervised learning works with raw, unlabeled data, where no target variable (Y) is given. In contrast, supervised learning uses labeled data where a classification label or a continuous target variable is provided for each example.

2. The goal of clustering is to find groups where data points within a single group are similar to one another and different from data points in other groups. This is achieved by minimizing intra-cluster distances (distances within a cluster) and maximizing inter-cluster distances (distances between different clusters).

3. Partitional algorithms construct a partition of n objects into a set of K clusters, where the user must specify the number of clusters K. They typically start with a random partitioning and refine it iteratively until a stopping condition is met.

4. The K-means algorithm follows these steps: 1) Decide on a value for k. 2) Initialize k cluster centers randomly. 3) Assign each object to its nearest cluster centroid. 4) Re-estimate the cluster centers based on the new memberships. 5) Repeat steps 3 and 4 until no objects change membership.

5. A centroid is the center of gravity or mean of all data points belonging to a cluster. In each iteration of the K-means algorithm, after data points are assigned to their nearest centroid, each centroid's position is updated by "jumping" to the mean location of the points it now "owns".

6. "Elbow finding" is a heuristic method to determine the right number of clusters (k). It involves running K-means for a range of k values and plotting the objective function (sum of squared distances) for each k. The optimal k is often found at the "elbow" or "knee" of the plot, where the rate of decrease in the objective function value abruptly slows.

7. The initial selection of random seeds can significantly impact the results of K-means. Some seed selections can lead to a poor convergence rate or cause the algorithm to converge to a sub-optimal clustering solution. To mitigate this, it is very important to try multiple starting points or use heuristics to select good seeds.

8. The mathematical objective is to minimize the sum of squared distances between each data point and the centroid of its assigned cluster. The algorithm converges because each reassignment step monotonically decreases this objective function, and it will eventually reach a fixed point where cluster memberships no longer change.

9. A Gaussian Mixture Model (GMM) is a partitional clustering method that assumes data is generated from a mixture of Gaussian distributions. It uses the Expectation-Maximization (EM) algorithm to iteratively revise each point's proportional membership to clusters and update each cluster's mean (centroid) and covariance (shape).

10. Purity is an external evaluation measure that assesses clustering quality with respect to ground truth (gold standard classes). For a single cluster, it is calculated as the ratio between the number of objects from the dominant class within that cluster and the total size of the cluster.

---

## Essay Questions

Develop a detailed response to the following prompts, synthesizing information from the source material.

1. Compare and contrast the K-means algorithm with the Gaussian Mixture Model (GMM) approach to partitional clustering. Discuss their core assumptions, iterative processes, and how they define cluster membership for data points.

2. Discuss the computational complexity and potential challenges of the K-means algorithm. How does its time complexity scale with the number of objects, dimensions, clusters, and iterations, and what are the primary strategies for overcoming its sensitivity to initialization?

3. Explain the mathematical objective function that the K-means algorithm seeks to optimize. Detail how the two main iterative steps—assigning memberships and updating centroids—work in conjunction to minimize this function and guarantee convergence.

4. Describe the methods for evaluating the quality of a clustering result. Differentiate between internal and external criteria, providing a specific example of an external criterion like "purity" and explaining how it is calculated and what it signifies.

5. Imagine you are tasked with clustering an unlabeled tabular dataset. Outline a complete workflow using the K-means methodology, from defining the problem to evaluating the final output. Your plan should address data representation, determining the value of K, the iterative clustering process, and dealing with potential convergence issues.

---

## Glossary of Key Terms

| Term | Definition |
|------|------------|
| **Centroid** | The center of gravity or mean of the data points assigned to a specific cluster. In K-means, centroids are iteratively re-calculated and updated. |
| **Clustering** | An unsupervised learning task aimed at finding groups (clusters) in data such that points within a group are similar to one another, while points in different groups are dissimilar. |
| **Covariance** | In the context of Gaussian Mixture Models (GMM), covariance defines the shape and orientation of the Gaussian distribution for a specific cluster. |
| **Elbow Method** | A heuristic for finding the optimal number of clusters (K) by plotting the value of the objective function for different values of K and identifying the "elbow" point where the rate of improvement sharply decreases. Also known as "knee finding". |
| **Expectation-Maximization (EM) Algorithm** | A general iterative procedure for finding maximum likelihood estimates of parameters in statistical models. K-means is considered a special case of the EM algorithm, which is known to converge. |
| **External Criterion** | A measure of clustering quality that assesses the result against a known ground truth or gold standard data. Purity is an example of an external criterion. |
| **Gaussian Mixture Model (GMM)** | A probabilistic model for clustering that assumes the data points are generated from a mixture of a finite number of Gaussian distributions with unknown parameters. |
| **Hierarchical Algorithms** | A class of clustering algorithms that create a nested sequence of clusters, either through a bottom-up (agglomerative) or top-down (divisive) approach. |
| **Inter-cluster Distance** | The measure of similarity or distance between different clusters. The goal of clustering is to maximize this distance. |
| **Internal Criterion** | A measure of clustering quality that relies only on the clustered data itself, such as evaluating for high intra-cluster similarity and low inter-cluster similarity. |
| **Intra-cluster Distance** | The measure of similarity or distance among data points within the same cluster. The goal of clustering is to minimize this distance. |
| **K-means** | A partitional clustering algorithm that aims to partition n observations into K clusters in which each observation belongs to the cluster with the nearest mean (centroid). |
| **Objective Function** | In K-means, this is the function the algorithm seeks to minimize. It is defined as the sum of the squared distances from each data point to its assigned cluster's centroid. |
| **Partitional Algorithms** | A class of clustering algorithms that divide a dataset into a set of K non-overlapping clusters, where K is specified by the user. They typically start with a random partition and refine it iteratively. |
| **Purity** | An external measure of cluster quality. For a given cluster, it is the ratio of the count of the most frequent class (from the ground truth) to the total number of data points in that cluster. |
| **Seed Choice** | The initial placement of the K cluster centers. This choice can be random and can significantly affect the final clustering result and convergence rate. |
| **Time Complexity** | A measure of the computational cost of an algorithm. For K-means, it is O(lKnp), where l is the number of iterations, K is the number of clusters, n is the number of data points, and p is the number of dimensions. |
| **Unsupervised Learning** | A type of machine learning that involves learning from raw, unlabeled, or unannotated data, without a given output variable (Y). |
