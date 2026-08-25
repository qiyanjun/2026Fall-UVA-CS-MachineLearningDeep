---
layout: post
title:  Algebra Review
lecture: S0-AlgbReview
lectureVersion: current
extraContent: S0-linalg-extra.pdf
notes: <a href="https://www.khanacademy.org/math/multivariable-calculus">Khanacademy Math</a> +  <a href="http://www.cs.cmu.edu/~zkolter/course/15-884/linalg-review.pdf">CMU study note</a>
video:  <a href="https://github.com/jakevdp/PythonDataScienceHandbook/tree/master/notebooks">Python Data Science Code tutorials</a>
categories: prerequisite 
background: true
tags:
- 1Basic
- background
---


# Comprehensive Linear Algebra and Matrix Calculus Review

## Study Guide

This study guide is designed to reinforce your understanding of fundamental concepts in linear algebra and matrix calculus, as presented in the source material. Mastering these topics is crucial for advanced studies in machine learning and related fields.

## I. Fundamental Definitions

- **Scalar**: Understand what a scalar is and how it's denoted.
- **Vector**: Define a vector, differentiate between row and column vectors, and understand its representation in real space (R^n).
- **Matrix**: Define a matrix, understand its dimensions (m-by-n), how its entries are denoted, and the concept of a square matrix.

## II. Special Matrices

- **Identity Matrix (I)**: Understand its structure and significance.
- **Diagonal Matrix**: Recognize its form.
- **Upper/Lower Triangular Matrices**: Identify these structures.
- **Symmetric Matrix**: Understand the property that defines a symmetric matrix.
- **Orthogonal Matrix**: Define an orthogonal matrix and its key property related to its inverse.

## III. Matrix Operations

### Transposition
- **Concept**: "Flipping" rows and columns.
- **Notation**: A^T.
- **Application**: Understand how to transpose a given matrix.

### Addition and Subtraction
- **Condition**: Matrices must be of the same size.
- **Process**: Entry-wise operation.
- **Examples**: Be able to perform addition and subtraction.

### Multiplication
- **Notation**: AB (pre-multiplying B by A, post-multiplying A by B).
- **Conformability**: Understand the condition for multiplication (number of columns in premultiplier must equal number of rows in postmultiplier).
- **Dimensions**: Predict the dimensions of the resulting matrix (m × n * n × p = m × p).
- **Non-Commutativity**: Recognize that AB ≠ BA, generally.
- **Associativity**: Understand A(BC) = (AB)C.
- **Properties**: (AB)^T = B^T A^T; multiplication with Identity Matrix.

### Special Uses
- **Scalar & Matrix Product**: Understand bA = Ab.
- **Dot (Inner) Product of Vectors**: a^T b (result is a scalar).
- **Outer Product of Vectors**: ab^T (result is a matrix).
- **Sum of Squared Elements of a Vector**: a^T a (useful for L2 norm).

### Norm (of Vector)
- **Concept**: Measure of "length" of a vector.
- **Common Norms**: L1, L2 (Euclidean), L∞.
- **L2 Norm (Euclidean)**: ||x|| = √(x^T x).
- **General Definition**: Understand the properties a function must satisfy to be a norm.

### Matrix Inversion
- **Notation**: A^(-1) or inv A.
- **Definition**: AA^(-1) = I = A^(-1)A.
- **Nonsingular vs. Singular**: Differentiate between matrices that have an inverse and those that don't.
- **Determinant Condition**: An inverse exists if and only if |A| ≠ 0.
- **Properties**: (AB)^(-1) = B^(-1)A^(-1); (A^T)^(-1) = (A^(-1))^T; (A^(-1))^(-1) = A.
- **Special Cases**: Inverse of diagonal matrices, orthogonal matrices (A^(-1) = A^T).
- **Pseudo-inverse (Extra)**: Basic understanding of its existence for non-square matrices.

### Matrix Rank
- **Linear Independence**: Define linearly independent vectors.
- **Definition of Rank**: Maximal number of linearly independent columns (or rows).
- **Properties**: rank(A) ≤ min(m,n). Full row rank, full column rank.
- **Determinant Connection**: Rank is the dimension of the largest square sub-matrix with a non-zero determinant.
- **Relationship to Singular Matrices**: How rank relates to singularity.
- **Rank of Product**: rank(AB) ≤ min(rank(A), rank(B)).

### Matrix Calculus
- **Review of Derivatives**: Understand the basic definition of a derivative for single-variable functions.
- **Multivariate Calculus Concepts**:
  - **Partial Derivative**: Extension of derivative to functions of multiple variables.
  - **Gradient**: Vector of partial derivatives (first derivative to gradient).
  - **Hessian Matrix**: Matrix of second partial derivatives (second derivative to Hessian).
  - **Denominator Layout**: Understand that the gradient's size is the same as the variable's size.
  - **Examples**: Be able to calculate simple gradients.

## IV. Additional Concepts (Mentioned as Extra or for Future Reference)

- **Trace()**: Sum of diagonal elements (not detailed in this source, but mentioned as "must know").
- **Eigenvalue / Eigenvectors**: Important for future topics.
- **Positive Definite Matrix, Gram Matrix, Quadratic Form, Projection**: Future topics.
- **Khan Academy**: A recommended resource for review.
- **Linear Transformation and Determinant**: Connection to matrix determinant and Jacobian determinant.
- **Laplacian of a function**: Trace of the Hessian.
- **Harmonic function**: Function whose Laplacian is 0.

## Quiz: Short-Answer Questions

1. What is the primary difference between a scalar and a vector in linear algebra notation?
2. Define a "column vector" and provide an example of a vector in R^4.
3. Explain the condition for two matrices to be "conformable" for multiplication. Why is this condition important?
4. Given matrices A (3×2) and B (2×4), what are the dimensions of the resulting matrix C = AB? What about C = BA?
5. State two key properties of matrix multiplication that distinguish it from scalar multiplication.
6. Describe the concept of a "norm" of a vector. What does the L2 norm represent?
7. What is the definition of the inverse of a matrix A (denoted A^(-1))? Under what crucial condition does a matrix inverse exist?
8. Differentiate between a "nonsingular" and a "singular" matrix.
9. Explain the concept of "linear independence" in the context of vectors. How does this relate to the rank of a matrix?
10. What is the gradient of a multivariate function? How does it relate to partial derivatives?

## Quiz Answer Key

1. **Scalar vs. Vector**: A scalar is a single number (e.g., 1 or 22), denoted with regular type. A vector is a single row or column of numbers (e.g., [1 2 3]), denoted with bold small letters.

2. **Column Vector and R^4 Example**: A column vector is a vector arranged vertically. An example of a vector in R^4 (a column vector) is v = (1,6,3,4)^T.

3. **Conformable Matrices**: For two matrices A and B to be conformable for multiplication (AB), the number of columns in the premultiplier (A) must equal the number of rows in the postmultiplier (B). This ensures that the dot product for each entry in the resulting matrix can be calculated.

4. **Matrix Multiplication Dimensions**: If A is (3×2) and B is (2×4), then C = AB will have dimensions (3×4). C = BA cannot be performed because the number of columns in B (4) does not equal the number of rows in A (3).

5. **Properties of Matrix Multiplication**:
   - Matrix multiplication is generally not commutative (AB ≠ BA).
   - Matrices must be conformable for multiplication to occur.

6. **Vector Norm**: A norm of a vector ||x|| is a measure of its "length." The L2 norm, also known as the Euclidean norm, represents the straight-line distance from the origin to the vector's endpoint (calculated as the square root of the sum of the squared elements).

7. **Matrix Inverse Definition and Condition**: The inverse of an n × n matrix A is the matrix A^(-1) such that AA^(-1) = I = A^(-1)A, where I is the identity matrix. A crucial condition for an inverse to exist is that the determinant of A (|A|) must not be zero.

8. **Nonsingular vs. Singular Matrix**: A "nonsingular" matrix is an n × n matrix that has an inverse. A "singular" matrix is an n × n matrix that does not have an inverse.

9. **Linear Independence and Rank**: A set of vectors is linearly independent if none of them can be written as a linear combination of the others. The rank of a matrix is defined as the maximal number of linearly independent columns (or rows) in that matrix.

10. **Gradient of a Multivariate Function**: The gradient of a multivariate function is a vector containing its partial derivatives with respect to each variable. It extends the concept of a derivative to functions of multiple variables, indicating the direction of the steepest ascent of the function.