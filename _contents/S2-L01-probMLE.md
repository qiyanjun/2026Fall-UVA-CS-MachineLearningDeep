---
LOrder: 160
layout: post
title: ProbReview + MLE  
lecture:  S2-MLE
lectureVersion: current
extraContent: S2-MLE 
video: <b>M1</b> <a href="https://youtu.be/RIvdfYIHT1I" target="_blank">Original</a> · <a href="https://youtu.be/l78hirhUWuM" target="_blank">AI-Clone</a> · <a href="https://youtu.be/pSSZ4bQircA" target="_blank">Author-Voice</a>
notes: 'Murphy (2022), <a href="https://probml.github.io/pml-book/book1.html" target="_blank">Probabilistic Machine Learning: An Introduction, Ch. 4: Statistics</a> · Brooks-Bartlett (2018), <a href="https://medium.com/data-science/probability-concepts-explained-maximum-likelihood-estimation-c7b4342fdbb1" target="_blank">Probability Concepts Explained: Maximum Likelihood Estimation</a>'
morenotes: <a href="http://statweb.stanford.edu/~susan/courses/s200/lectures/lect11.pdf"> MLE </a>  / <a href="https://medium.com/@rrfd/what-is-maximum-likelihood-estimation-examples-in-python-791153818030"> MLE code</a> + <a href="https://www.youtube.com/watch?v=aDW44NPhNw0&list=PLs8w1Cdi-zvY9ICoYqu1XV0YoTQgShXw2">Error Metrics</a>
categories: basics
background: true
tags:
- 1Basic
- Background
---


# Study Guide: Maximum Likelihood Estimation

## Quiz: Key Concepts Review

Answer the following questions in 2-3 sentences each based on the provided course material.

1. What is the fundamental goal of Machine Learning as described in the lecture?
2. Define "Sample Space" and "Event" in the context of probability theory, and provide an example for each.
3. What is the core principle of Maximum Likelihood Estimation (MLE)?
4. Why is it often more convenient to work with the log-likelihood function instead of the likelihood function itself?
5. For a Bernoulli distribution, what is the Maximum Likelihood Estimate for the parameter p (the probability of success)?
6. How does a probability density function (pdf) for a continuous random variable differ from a probability mass function (pmf) for a discrete random variable?
7. What do the mean and covariance matrix represent in a multivariate normal (Gaussian) distribution?
8. What key assumption connects Maximum Likelihood Estimation to the least squares error function in linear regression?
9. In the probabilistic interpretation of linear regression, what distribution is the error term (ε) assumed to follow?
10. Define the covariance between two random variables, X and Y.

## Essay Questions

The following questions are designed for longer-form, essay-style answers to test a deeper synthesis of the material. Answers are not provided.

1. Explain the complete process of deriving the Maximum Likelihood Estimate for the parameter p of a Bernoulli distribution. Detail the role of the likelihood and log-likelihood functions, the use of differentiation to find the maximum, and interpret the final result.

2. Describe the relationship between basic probability concepts (sample space, events, random variables) and the task of machine learning. How do these foundational ideas enable the formulation of a model and the application of an estimation technique like MLE?

3. Compare and contrast discrete and continuous random variables. Discuss their respective probability functions (pmf vs. pdf), how their means (expectations) are calculated, and provide an example of each as mentioned in the lecture notes.

4. Discuss the probabilistic interpretation of linear regression. Explain how assuming a Gaussian distribution for the error term allows the problem to be framed as a Maximum Likelihood Estimation task, and show why maximizing the log-likelihood is equivalent to minimizing the sum of squared errors.

5. What are the key properties of a multivariate Gaussian (Normal) distribution? Describe the roles of the mean vector and the covariance matrix, and explain how they influence the shape and orientation of the probability density function, using the bivariate normal distribution as an illustrative example.

## Answer Key

### Quiz Answers

1. The goal of Machine Learning is to optimize a performance criterion using example data or past experience. The primary aim is to develop models that can generalize their performance to new, unseen data.

2. A Sample Space (O) is the set of all possible outcomes of an experiment, such as {HH, HT, TH, TT} for tossing a coin twice. An Event is a subset of the sample space, such as the event of the first toss being a head, which would be the set {HH, HT}.

3. The principle of Maximum Likelihood Estimation is to select the set of model parameters (θ) that are most likely to have produced the observed data. This is achieved by assuming a particular model and finding the parameter values that maximize the joint probability (the likelihood) of the observed sample set.

4. Working with the log-likelihood function is convenient because it converts the joint probability, which is a product of individual probabilities, into a sum. This transformation from Π P(Zi|θ) to Σ log(P(Zi|θ)) simplifies the mathematical process of finding the maximum, particularly when taking derivatives.

5. The MLE for the parameter p in a Bernoulli distribution is the sample proportion of positive outcomes. It is calculated as p̂ = x/n, where x is the number of successes (e.g., heads) observed in n trials.

6. A probability mass function (pmf) gives the probability that a discrete random variable is equal to a specific value, P(X = xi). A probability density function (pdf) describes the probability density for a continuous variable, and the actual probability is obtained by taking the integral of the pdf over a given range.

7. In a multivariate normal distribution, the mean vector (μ) is the point where the PDF reaches its peak value. The covariance matrix (Σ) captures the linear dependencies among the variables and determines the shape and orientation of the elliptical contours of equal probability density.

8. The key assumption is that the error term, or residual, in the linear regression model is independent and identically distributed (IID) according to a Gaussian (Normal) distribution with a mean of zero. This assumption makes maximizing the data likelihood equivalent to minimizing the residual sum of squares.

9. In the probabilistic view of linear regression, the error term ε is assumed to follow a Gaussian distribution with a mean of 0 and some variance σ², denoted as N(0, σ).

10. Covariance measures the joint variability of two random variables. It is defined as the expectation of the product of their deviations from their individual means: Cov(X,Y) = E((X − µx)(Y − µy)).

## Glossary of Key Terms

| Term | Definition |
|------|------------|
| **Sample Space (O)** | The set of all possible outcomes of an experiment. |
| **Event** | A subset of the sample space O. |
| **Random Variable (RV)** | A function that maps outcomes from the sample space to an attribute space, providing a concise way of specifying attributes of outcomes. |
| **Discrete Random Variable** | A random variable that may take on only a countable number of distinct values. |
| **Continuous Random Variable** | A random variable described by a probability density function (pdf) rather than a probability mass function (pmf). |
| **Probability Density Function (pdf)** | A function f(x) that describes the probability density for a continuous random variable in terms of an input variable x. It must be non-negative, and its integral over all possible values is 1. |
| **Probability Mass Function (pmf)** | A function that gives the probability that a discrete random variable is exactly equal to some value. |
| **Maximum Likelihood Estimation (MLE)** | An estimation technique where one chooses a set of parameters that are most likely to have produced the observed results. It assumes a model with unknown parameters and maximizes the likelihood of the observed data with respect to those parameters. |
| **Likelihood Function** | The joint probability of observing a set of data, expressed as a function of the model parameters θ. For independent samples, it is the product of the individual probabilities: `P(Z₁...Zₙ \| θ) = Π P(Zᵢ \| θ)`. |
| **Log-Likelihood Function** | The natural logarithm of the likelihood function. It converts the product of probabilities into a sum: `log(L(θ)) = Σ log(P(Zᵢ \| θ))`. |
| **Bernoulli Distribution** | A probability distribution for a binary random variable that takes a value of 1 (success) with probability p and a value of 0 (failure) with probability 1-p. |
| **Binomial Distribution** | A discrete probability distribution describing the number of successes in a sequence of k independent Bernoulli trials, each with a success probability of p. |
| **Gaussian (Normal) Distribution** | A widely used continuous probability distribution for a real-valued random variable, characterized by its mean (μ) and variance (σ²). |
| **Mean (Expectation)** | A measure of the central tendency of a random variable. For a discrete RV, it's Σ vᵢ * P(X=vᵢ); for a continuous RV, it's ∫ x * f(x)dx. |
| **Variance** | A measure of the spread of a random variable around its mean. It is defined as Var(X) = E((X − µ)²). |
| **Covariance** | A measure of the joint variability of two random variables, X and Y. It is defined as Cov(X,Y) = E((X − µx)(Y − µy)). |
| **Correlation** | A normalized measure of the linear relationship between two random variables, calculated as ρ(X,Y) = Cov(X,Y) / (σx * σy). Its value ranges from -1 to 1. |