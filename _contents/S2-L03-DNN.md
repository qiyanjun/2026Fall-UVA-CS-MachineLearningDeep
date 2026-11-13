---
LOrder: 190
layout: post
title: NN and Deep Learning
lecture: S2-L12-deepNNBasics
lectureVersion: current
video: <b>M1</b> <a href="https://youtu.be/BW9aQ0uCWJI" target="_blank">Original</a> · <a href="https://youtu.be/ReLB9nc45lY" target="_blank">AI-Clone</a> · <a href="https://youtu.be/35alpWC1dK8" target="_blank">Author-Voice</a>  |  <b>M2</b> <a href="https://youtu.be/rxwDm3R6gp0" target="_blank">Original</a> · <a href="https://youtu.be/ol2tqQetIcQ" target="_blank">AI-Clone</a> · <a href="https://youtu.be/CFz6gOW9YvI" target="_blank">Author-Voice</a>  |  <b>M3</b> <a href="https://youtu.be/viVhuJmCobI" target="_blank">Original</a> · <a href="https://youtu.be/YJDBxVdX3E8" target="_blank">AI-Clone</a> · <a href="https://youtu.be/4648Si8UwrU" target="_blank">Author-Voice</a>  |  <b>M4</b> <a href="https://youtu.be/-B37QghuoL8" target="_blank">Original</a> · <a href="https://youtu.be/i9JgfAJqGh4" target="_blank">AI-Clone</a> · <a href="https://youtu.be/U9uIS4AB__M" target="_blank">Author-Voice</a>
notes: 'Goodfellow, Bengio & Courville (2016), <a href="https://www.deeplearningbook.org/contents/mlp.html" target="_blank">Deep Learning, Ch. 6: Deep Feedforward Networks</a> · Nielsen (2015), <a href="http://neuralnetworksanddeeplearning.com/chap1.html" target="_blank">Neural Networks and Deep Learning, Ch. 1</a>'
morenotes: <a href="https://github.com/afshinea/stanford-cs-230-deep-learning"> DNN Cheatsheets </a>
categories: structured
tags:
- 3Classification
- Nonlinear
- Deep
- Discriminative
---


# Neural Networks and Backpropagation: A Study Guide


---

## Quiz: Short-Answer Questions

Answer each question in 2-3 sentences based on the provided source material.

1. How can a single neuron be conceptualized as an extension of logistic regression?
2. What was the historical Perceptron, and what type of activation function did it use?
3. Describe the structure of a Multi-Layer Perceptron (MLP). What makes a neural network "deep"?
4. What is the purpose of the Softmax function, particularly in the context of multi-class classification?
5. Identify two different types of loss functions mentioned in the text and the tasks they are suited for.
6. What is the core purpose of the backpropagation algorithm in training a neural network?
7. Briefly describe the two main phases of the training loop for a neural network using backpropagation.
8. What problem can occur with gradient magnitudes during training, and why is divergence considered the worse outcome?
9. What is "Dropout" and how does it function as a regularization technique?
10. Explain the concept of Batch Normalization and its benefits for the training process.
11. What role does the bias term play in a neuron?
12. Show that logistic regression is a special case of softmax for two classes. Why is softmax over-parameterized?
13. How is the binary cross-entropy loss derived from the Bernoulli likelihood, and what is the gradient of softmax + cross-entropy with respect to the scores z<sub>i</sub>?
14. In the lecture's backpropagation example (z<sub>1</sub> = x<sub>1</sub>w<sub>1</sub> + x<sub>2</sub>w<sub>3</sub> + b<sub>1</sub>, h<sub>1</sub> = sigmoid(z<sub>1</sub>), ŷ = h<sub>1</sub>w<sub>5</sub> + h<sub>2</sub>w<sub>6</sub> + b<sub>3</sub>, E = (y − ŷ)<sup>2</sup>), how do you compute ∂E/∂w<sub>5</sub> and ∂E/∂w<sub>1</sub>?
15. How should a network trained with dropout make predictions at test time?

---

## Answer Key

1. **A single neuron can be seen as an expanded logistic regression unit.** It performs a two-stage process: first, a summing function calculates a weighted sum of the inputs plus a bias (z = wᵀ · x + b), and second, a non-linear activation function (like sigmoid) "squashes" the result into a desired range.

2. **The Perceptron, first proposed by Rosenblatt in 1958, was a simple one-neuron unit used to classify input into one of two categories.** It used a step function as its activation function, which outputs +1 if the input is greater than or equal to zero and -1 otherwise.

3. **A Multi-Layer Perceptron (MLP), or a feed-forward neural network, is composed of an input layer, one or more hidden layers, and an output layer.** A neural network is considered "deep" when it contains many hidden layers.

4. **The Softmax function is a normalizing function used as the final layer for multi-class classification.** It converts the output of each class unit into a probability, ensuring that the sum of all output probabilities is equal to 1.

5. **The text describes Sum of Squared Errors (SSE) loss, which is suitable for regression tasks, calculating (y₁-ŷ₁)²+(y₂-ŷ₂)².** It also details the Cross-Entropy loss function (also called negative log-likelihood), which is used for binary and multi-class classification tasks.

6. **The backpropagation algorithm is used to learn the optimal weights for a neural network by jointly optimizing all parameters.** It accomplishes this by calculating the gradient of the loss function with respect to each weight in the network, even those in lower layers.

7. **The first phase is the "Forward" pass, where inputs are fed through the network layer by layer to compute the final output and the loss.** The second phase is the "Backward" pass, where the algorithm propagates local gradients from the final loss function back through the network to calculate each layer's gradient for weight updates.

8. **During training, gradients can become too big, leading to divergence, or too small, leading to slow convergence.** Divergence is considered much worse because the training process fails to find a solution, whereas slow convergence simply takes more time.

9. **Dropout is a regularization technique where some neurons are randomly set to zero during the forward pass of training.** This process is akin to training a large ensemble of models that share parameters, which helps prevent overfitting.

10. **Batch Normalization is a technique that standardizes the activations of a prior layer, which stabilizes and speeds up training.** It improves gradient flow, allows for higher learning rates, reduces dependence on initialization, and acts as a form of regularization.

11. **The bias term b adds a constant to the weighted sum, z = wᵀ · x + b, by attaching a weight to a constant input of +1.** Without it, y = σ(w<sub>1</sub>x<sub>1</sub> + ... + w<sub>n</sub>x<sub>n</sub>), so z is always 0 at the origin; with the bias w<sub>0</sub>, the activation (and the decision boundary) can shift away from the origin.

12. **For two classes, softmax gives y<sub>1</sub> = e<sup>z<sub>1</sub></sup> / (e<sup>z<sub>1</sub></sup> + e<sup>z<sub>0</sub></sup>) = 1 / (1 + e<sup>−(z<sub>1</sub> − z<sub>0</sub>)</sup>), which is the logistic (sigmoid) function of z<sub>1</sub> − z<sub>0</sub>.** Adding the same constant to both z<sub>1</sub> and z<sub>0</sub> has no effect, because the probabilities must add to 1; the logistic binary case avoids these redundant parameters.

13. **Model y with a Bernoulli distribution, p(y \| x) = ŷ<sup>y</sup>(1 − ŷ)<sup>1−y</sup>, and take the negative log-likelihood: E = −{y log ŷ + (1 − y) log(1 − ŷ)}.** This is the cross-entropy loss, also called "deviance". For K classes with a softmax output, E = −Σ<sub>j</sub> y<sub>j</sub> ln ŷ<sub>j</sub>, and its gradient is ∂E/∂z<sub>i</sub> = ŷ<sub>i</sub> − y<sub>true,i</sub>: the error between the predicted output and the true label.

14. **Treat the network as a chain of differentiable functions f<sub>4</sub>(f<sub>3</sub>(f<sub>2</sub>(f<sub>1</sub>(·)))) and multiply local gradients with the chain rule.** For the top layer, ∂E/∂w<sub>5</sub> = ∂E/∂ŷ · ∂ŷ/∂w<sub>5</sub> = −2(y − ŷ) · h<sub>1</sub>. For the lower layer, ∂E/∂w<sub>1</sub> = ∂E/∂ŷ · ∂ŷ/∂h<sub>1</sub> · ∂h<sub>1</sub>/∂z<sub>1</sub> · ∂z<sub>1</sub>/∂w<sub>1</sub> = −2(y − ŷ) · w<sub>5</sub> · h<sub>1</sub>(1 − h<sub>1</sub>) · x<sub>1</sub>, using the sigmoid derivative σ(1 − σ). The factor ∂E/∂ŷ is already computed from the layer above and is reused, which is why backpropagation works backward layer by layer (as in the Module IV binary-classification walk-through x → W<sub>1</sub> → z<sub>1</sub> → h<sub>1</sub> → w<sub>2</sub> → z<sub>2</sub> → ŷ).

15. **Ideally, the noise from dropout is integrated out at test time.** A Monte Carlo approximation does many forward passes with different dropout masks and averages all the predictions, consistent with viewing dropout as training a large ensemble of models that share parameters (each binary mask is one model).

---

## Essay Questions

Consider the following questions to synthesize the concepts from the course material. Formulate a detailed, evidence-based response for each.

1. **Explain the journey from Logistic Regression as a linear classifier to a Multi-Layer Perceptron capable of modeling non-linear decision boundaries.** How do hidden layers and non-linear activation functions enable this capability?

2. **Describe the complete process of training a neural network for multi-class classification using Mini-batch Stochastic Gradient Descent (SGD) and backpropagation.** Detail the roles of the forward pass, the Softmax layer, the cross-entropy loss function, the backward pass, and the weight update step.

3. **Compare and contrast the activation functions presented on the lecture's activation-functions slide:** Sigmoid, tanh, ReLU max(0, x), Leaky ReLU max(0.1x, x), Maxout, and ELU. Discuss their mathematical forms, their derivatives, and the potential implications of these properties on the training process.

4. **Discuss the importance of proper network initialization and regularization in deep learning.** Using the concepts of Xavier initialization, Batch Normalization, and Dropout, explain how practitioners can avoid common training pitfalls like poor gradient flow and overfitting.

5. **From a "block view," a neural network can be seen as a series of differentiable, parameterized functions.** Explain how the chain rule of calculus is the fundamental mathematical principle behind backpropagation, allowing gradients to be passed "backward" through these blocks to update parameters in the earliest layers of the network.

---

## Glossary of Key Terms

| Term | Definition |
|------|------------|
| **Activation Function** | A function applied to the output of a neuron (the weighted sum of inputs) to introduce non-linearity into the network. Also known as a transfer function. Examples include Sigmoid, tanh, and ReLU. |
| **Backpropagation** | An algorithm for training neural networks by using the chain rule to efficiently compute the gradients of the loss function with respect to all the weights and biases in the network. It involves a forward pass to compute outputs and loss, and a backward pass to propagate gradients. |
| **Backpropagation chain rule** | Each block passes back its gradient times its local gradient (output w.r.t. input); for lower layers the gradients from the layers above are already computed and reused. |
| **Batch Normalization** | A technique to normalize the activations of a prior layer, which improves gradient flow, allows higher learning rates, and acts as a form of regularization. |
| **Bias Term (b)** | A constant value added to the product of inputs and weights in a neuron. It allows the activation function to be shifted to the left or right, which is critical for learning. |
| **Cross-Entropy Loss** | A loss function used for classification tasks. It measures the performance of a classification model whose output is a probability value between 0 and 1. Also known as negative log-likelihood or deviance. |
| **Deep Neural Network (DNN)** | A neural network with many hidden layers between the input and output layers. |
| **Bernoulli likelihood** | p(y \| x) = ŷ<sup>y</sup>(1 − ŷ)<sup>1−y</sup>; its negative log gives the binary cross-entropy loss −{y log ŷ + (1 − y) log(1 − ŷ)}. |
| **Dropout** | A regularization technique where randomly selected neurons are ignored ("dropped out") during training. This prevents units from co-adapting too much and helps prevent overfitting. |
| **Dropout at test time** | Ideally integrate out the dropout noise; a Monte Carlo approximation averages the predictions of many forward passes with different dropout masks. |
| **Feed-Forward Neural Network** | A type of neural network where connections between the nodes do not form a cycle. Information moves in only one direction: from the input nodes, through the hidden nodes, and to the output nodes. The Multi-Layer Perceptron (MLP) is a feed-forward network. |
| **Loss Function** | A function that computes a single number representing the "cost" or "error" associated with the network's prediction (ŷ) versus the true label (y). The goal of training is to minimize this value. |
| **Multi-Layer Perceptron (MLP)** | A class of feed-forward artificial neural network. An MLP consists of at least three layers of nodes: an input layer, one or more hidden layers, and an output layer. |
| **Neuron** | The fundamental processing unit of a neural network. It takes multiple inputs, computes a weighted sum, adds a bias, and then passes the result through an activation function. |
| **Perceptron** | A single-neuron unit proposed by Rosenblatt (1958) that uses a step function for activation. It is a linear classifier. |
| **Rectified Linear Unit (ReLU)** | An activation function defined as h(x) = max(0, x). It is computationally efficient and has become a default activation function for many types of neural networks. Variations include Leaky ReLU. |
| **Sigmoid Function** | An activation function that squashes its input value into a range between 0 and 1. Its formula is h(x) = 1 / (1 + exp(-x)). |
| **Softmax Function** | A function that converts a vector of K real numbers into a probability distribution of K possible outcomes. It is often used as the last activation function of a neural network to normalize the output for multi-class classification. |
| **Softmax + cross-entropy gradient** | For a softmax output layer with cross-entropy loss, ∂E/∂z<sub>i</sub> = ŷ<sub>i</sub> − y<sub>true,i</sub>. For two classes, softmax reduces to the logistic (sigmoid) function of z<sub>1</sub> − z<sub>0</sub>. |
| **Stochastic Gradient Descent (SGD)** | An iterative optimization algorithm used to find the minimum of a loss function. In each iteration, it estimates the gradient based on a single training example or a small "mini-batch" of examples to update the network's weights. |
| **Sum of Squared Errors (SSE)** | A loss function used for regression tasks that measures the sum of the squared differences between the predicted values and the actual values. |
| **tanh Function** | An activation function that squashes its input value into a range between -1 and 1. Its formula is h(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x)). |
| **Weights (w)** | Parameters within a neural network that transform input data within the network's layers. The network learns by modifying its weights during training. |
| **Xavier Initialization** | A method for initializing the weights in a neural network to maintain a reasonable variance of activations and gradients across layers, which is important for stable training. |

