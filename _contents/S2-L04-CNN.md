---
LOrder: 190
layout: post
title: CNN 
lecture: S2-deepImageCNN
lectureVersion: current
video: <b>M1</b> <a href="https://youtu.be/foOw7tTDZq0" target="_blank">Original</a> · <a href="https://youtu.be/gj92q2R5foQ" target="_blank">AI-Clone</a> · <a href="https://youtu.be/BFacOv9zWPE" target="_blank">Author-Voice</a>  |  <b>M2</b> <a href="https://youtu.be/wwg3-8g-8no" target="_blank">Original</a> · <a href="https://youtu.be/c_0ym5SRYLs" target="_blank">AI-Clone</a> · <a href="https://youtu.be/JBwTZ09_78o" target="_blank">Author-Voice</a>
extraContent:  S2-dim-PCA
notes: 'Karpathy, Li & Johnson (2016), <a href="https://cs231n.github.io/convolutional-networks/" target="_blank">CS231n: Convolutional Neural Networks for Visual Recognition</a> · Olah (2014), <a href="https://colah.github.io/posts/2014-07-Conv-Nets-Modular/" target="_blank">Conv Nets: A Modular Perspective</a>'
morenotes: <a href="https://colab.research.google.com/github/qiyanjun/2026Fall-UVA-CS-MachineLearningDeep/blob/main/notebook/L13_keras_mnist_convnet.ipynb"> Keras notebook</a> + <a href="https://www.kaggle.com/code/shawon10/covid-19-diagnosis-from-images-using-densenet121"> another Keras Kaggle notebook</a> + <a href="https://github.com/qiyanjun/2026Fall-UVA-CS-MachineLearningDeep/blob/main/Lectures/S2-PCA-extra.pdf"> [PCA extra slides] </a>
categories: 2D (Vision)
tags:
- 3Classification
- Nonlinear
- Deep
- Discriminative
---


# Notebooks to run and experienc: 
- <a href="https://colab.research.google.com/github/qiyanjun/2026Fall-UVA-CS-MachineLearningDeep/blob/main/notebook/L13_keras_mnist_convnet.ipynb"> L13 Keras notebook</a>  
- <a href="https://www.kaggle.com/code/shawon10/covid-19-diagnosis-from-images-using-densenet121"> another Keras Kaggle notebook</a>



# Study Guide: Supervised Image Classification and Convolutional Neural Networks


---

## Quiz

Answer the following questions in 2-3 sentences each, based on the provided lecture context.

1. What were the primary challenges that led to the "Winter of Neural Networks" around the 2000s?
2. Explain the three key properties of images that Convolutional Neural Networks are specifically designed to leverage.
3. What is the core function of a "filter" in a convolutional layer, and what property of images does it relate to?
4. Describe the concept of "translation invariance" and explain how a CNN achieves it.
5. What is the purpose of the Max Pooling operation in a CNN?
6. How does a convolutional layer differ from a fully-connected layer in terms of parameter usage?
7. What are the three main components of a dataset used for model selection and assessment in a "data-rich scenario"?
8. According to the lecture, who invented the CNN and in what year was the foundational paper published?
9. What happens to the output of the convolutional and max pooling layers before it is fed into a fully connected feedforward network?
10. In the context of a tabular dataset for classification, what are the three main types of columns identified in the lecture?

---

## Answer Key

1. **The "Winter of Neural Networks" was caused by several challenges.** The optimization problem was non-convex, requiring many tricks for tuning hyperparameters like the number of layers and hidden units. Additionally, these networks were hard to analyze theoretically, and large labeled datasets were rare during that period.

2. **CNNs leverage three key properties of images.** The first is Locality, where patterns are smaller than the whole image. The second is Translation Invariance, meaning the same patterns can appear in different regions. The third is that Subsampling the pixels will not change the object, allowing the image to be made smaller.

3. **A filter in a convolutional layer is a matrix of parameters (weights) designed to detect a small, specific pattern in the input image, such as an edge or a curve.** This directly relates to the property of Locality, as the filter does not need to see the whole image to discover its target pattern.

4. **Translation invariance is the principle that an object's appearance is independent of its location within an image.** A CNN achieves this through weight sharing, where the same set of filter parameters is applied across all regions of the image to detect a specific pattern regardless of where it appears.

5. **The purpose of the Max Pooling operation is to subsample the feature map created by the convolutional layer.** It reduces the spatial dimensions of the data, which makes the image smaller and results in fewer parameters for the network to process in subsequent layers.

6. **A convolutional layer uses significantly fewer parameters than a fully-connected layer.** This is because its neurons are only connected to a small local region of the input (locality) and it reuses the same parameters across the entire image (weight sharing), whereas every neuron in a fully-connected layer connects to every input from the previous layer.

7. **In a data-rich scenario, the dataset is split into three parts for model selection and assessment.** These are the Training set, the Validation set, and the Test set.

8. **The Convolutional Neural Network (CNN) was invented by Professor Yann LeCun.** The foundational paper, "Gradient-based learning applied to document recognition," was published in 1998.

9. **After the final max pooling layer, the resulting multi-channel, smaller image (a 3-D tensor) is processed by a Flatten operation.** This converts the 3-D data into a one-dimensional vector, which can then be used as input for the subsequent fully connected feedforward network.

10. **For a tabular dataset, the columns are identified as Features (also called attributes, predictors, or independent variables), which are all columns except the last.** The last column is the Target (also called outcome, label, or dependent variable), which is the special column to be predicted. The rows are referred to as Data points or instances.

---

## Essay Questions

Develop a detailed, essay-format response for each of the following prompts. No answers are provided for this section.

1. **Trace the historical evolution of artificial intelligence and machine learning from Alan Turing's 1950 paper to the landmark 2012 paper on ImageNet classification.** Discuss the key milestones, influential figures, dominant technologies of each era (e.g., expert systems, SVMs), and the factors that contributed to the "Winter of Neural Networks."

2. **Provide a comprehensive explanation of the complete architecture of a standard CNN for image classification, as described in the lecture.** Detail the journey of an input image through the network, explaining the purpose and mechanics of the Convolution layer, Max Pooling layer, Flatten operation, and the final Fully Connected network with softmax.

3. **Elaborate on the three fundamental properties of images that make CNNs more effective than traditional Multilayer Perceptrons (MLPs) for image-related tasks.** For each property (Locality, Translation Invariance, and Subsampling), explain what it is and which specific CNN mechanism (e.g., filters, weight sharing, pooling) is designed to exploit it.

4. **Compare and contrast the processes of "Model Selection" and "Model Assessment."** Describe the different strategies available for these processes, such as the train/validation/test split for data-rich scenarios and methods like Cross-Validation for when data is insufficient.

5. **Using the provided example of a 6x6 image and a 3x3 filter, explain the mathematical process of convolution.** Describe how applying the filter across the image generates a "feature map" and how parameters like stride can alter the dimensions of this output.

---

## Glossary of Key Terms

| Term | Definition |
|------|------------|
| **Convolution** | An operation where a filter (a small matrix of parameters) is applied across an input image to produce a feature map. This process leverages the properties of locality and translation invariance. |
| **Convolutional Neural Networks (CNN)** | A type of neural network invented by Yann LeCun, first successfully trained with many layers in 1998. It is specifically designed for data with grid-like topologies, such as 2D images, by leveraging properties of locality, translation invariance, and subsampling. |
| **Cross-Validation (k-CV)** | A method for model selection and assessment used when there is insufficient data to split into three parts. It involves efficiently reusing samples to choose hyperparameters. |
| **Feature Map** | The output of applying a filter across an image in a convolutional layer. Each feature map corresponds to a specific filter and represents the detection of that filter's pattern across the input. |
| **Features** | In a tabular dataset, these are the columns used as predictors or independent variables to predict the target. They are also referred to as attributes, dimensions, covariates, or regressors. |
| **Filter** | A small matrix of learnable parameters (weights) used in a convolution layer to detect specific patterns (e.g., edges, textures) in an input image. |
| **Flatten** | An operation in a CNN that converts the multi-dimensional output of the convolutional/pooling layers into a single one-dimensional vector. This vector is then fed into a fully connected network. |
| **Fully Connected Feedforward Network** | The final part of a typical CNN architecture. It takes the flattened vector as input and performs classification, often ending with a softmax layer to output probabilities for each class. |
| **Locality** | A property of images where patterns of interest (like a bird's beak) are much smaller than the whole image. CNNs exploit this by using small filters, meaning a neuron does not have to see the entire image to discover a pattern. |
| **Max Pooling** | A subsampling technique where a feature map is downsized by taking the maximum value over a defined window. This reduces the number of parameters and computational complexity, making the representation more robust to small shifts. |
| **Model Assessment** | The process of, having chosen a final model, estimating its prediction error on new, unseen data. |
| **Model Selection** | The process of estimating the performance of different models (or models with different hyperparameters) in order to choose the best one. This is also referred to as hyperparameter tuning. |
| **Subsampling** | The property that reducing the resolution of an image (e.g., by taking every other pixel) will not fundamentally change the object depicted. This is the principle behind pooling layers in a CNN. |
| **Target** | In a tabular dataset, this is the special column to be predicted. It is also referred to as the outcome, response, label, or dependent variable. |
| **Translation Invariance** | A property of images where the appearance of an object is independent of its location. CNNs model this by sharing weights, meaning the same filter is used to detect a pattern regardless of where it appears in the image. |
| **Weight Sharing** | The practice in a CNN where the same set of filter parameters is applied across all spatial locations in an input image. This drastically reduces the total number of parameters and enforces translation invariance. |
