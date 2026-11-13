---
LOrder: 290
layout: post
title: CNN 
lecture: S2-L13-deepImageCNN
lectureVersion: current
video: <b>M1</b> <a href="https://youtu.be/foOw7tTDZq0" target="_blank">Original</a> · <a href="https://youtu.be/gj92q2R5foQ" target="_blank">AI-Clone</a> · <a href="https://youtu.be/BFacOv9zWPE" target="_blank">Author-Voice</a>  |  <b>M2</b> <a href="https://youtu.be/wwg3-8g-8no" target="_blank">Original</a> · <a href="https://youtu.be/c_0ym5SRYLs" target="_blank">AI-Clone</a> · <a href="https://youtu.be/JBwTZ09_78o" target="_blank">Author-Voice</a>
extraContent: 
notes: 'Karpathy, Li & Johnson (2016), <a href="https://cs231n.github.io/convolutional-networks/" target="_blank">CS231n: Convolutional Neural Networks for Visual Recognition</a> · Olah (2014), <a href="https://colah.github.io/posts/2014-07-Conv-Nets-Modular/" target="_blank">Conv Nets: A Modular Perspective</a>'
# morenotes: <a href="https://colab.research.google.com/github/qiyanjun/2026Fall-UVA-CS-MachineLearningDeep/blob/main/notebook/L13_pytorch_mnist_convnet.ipynb"> PyTorch notebook</a> + <a href="https://colab.research.google.com/github/qiyanjun/2026Fall-UVA-CS-MachineLearningDeep/blob/main/notebook/L13_keras_mnist_convnet.ipynb"> Keras version (same model)</a> + <a href="https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html"> PyTorch ResNet-18 transfer-learning tutorial</a>
categories: 2D (Vision)
tags:
- 3Classification
- Nonlinear
- Deep
- Discriminative
notebooks: '<a href="https://colab.research.google.com/github/qiyanjun/2026Fall-UVA-CS-MachineLearningDeep/blob/main/notebook/L13_pytorch_mnist_convnet.ipynb" target="_blank">notebook/L13_pytorch_mnist_convnet.ipynb</a> · <a href="https://colab.research.google.com/github/qiyanjun/2026Fall-UVA-CS-MachineLearningDeep/blob/main/notebook/L13_keras_mnist_convnet.ipynb" target="_blank">notebook/L13_keras_mnist_convnet.ipynb (Keras version, same model)</a> · <a href="https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html" target="_blank">PyTorch tutorial: Transfer learning with ResNet-18 (Run in Google Colab on the page)</a>'
---

### In this lecture, we cover:
- Background: early history of AI/ML (Turing 1950, Dartmouth 1956, expert systems, CNN/RNN in the 1990s, SVMs in the 2000s) and the "Winter of Neural Networks"
- Review: model selection and assessment
  + Pipelines: (1) train / validation / test; (2) k-CV on train to choose hyperparameters, then test
  + Confusion matrix (TP, TN, FP, FN) and metrics: accuracy, precision, recall (sensitivity), specificity, F1
  + Deep learning frameworks (PyTorch, TensorFlow, Keras, ...) and a PyTorch sample model
- Convolutional neural networks (LeCun et al. 1998; ImageNet, Krizhevsky et al. 2012)
  + Three image properties: locality, translation invariance, subsampling
  + Convolution: 3×3 filters on a 6×6 image, stride, feature maps; each filter is a channel
  + Convolution vs. fully connected: 2 filters need 3×3×2 = 18 parameters vs. 36×2 = 72
  + Max pooling vs. average pooling, and what happens when the image shifts one pixel
  + Color images as 3-channel (R, G, B) tensors; flatten, then a fully connected network with softmax
  + CNN in Keras: 1×28×28 → 25×26×26 → 25×13×13 → 50×11×11 → 50×5×5 → flatten to 1250; 9 and 225 parameters per filter
- More applications and architectures
  + Playing Go (19×19 board as an image) and speech (filters move along the frequency axis of a spectrogram)
  + Accuracy vs. compute of ImageNet models; the residual trick and ResNet; EfficientNet

# Notebooks to run and experiment with:
- <a href="https://colab.research.google.com/github/qiyanjun/2026Fall-UVA-CS-MachineLearningDeep/blob/main/notebook/L13_pytorch_mnist_convnet.ipynb"> L13 PyTorch notebook</a>  
- <a href="https://colab.research.google.com/github/qiyanjun/2026Fall-UVA-CS-MachineLearningDeep/blob/main/notebook/L13_keras_mnist_convnet.ipynb"> L13 Keras version (same model)</a>  
- <a href="https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html"> PyTorch ResNet-18 transfer-learning tutorial</a>



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
7. What two pipelines for model selection (hyperparameter tuning) and model assessment does the lecture review?
8. According to the lecture, who invented the CNN and in what year was the foundational paper published?
9. What happens to the output of the convolutional and max pooling layers before it is fed into a fully connected feedforward network?
10. Define TP, TN, FP, and FN, and give the formulas for precision and recall.
11. In the lecture's one-pixel-shift example, how do max pooling and average pooling respond, and why?
12. How is a color image represented as CNN input, and how does that change the filters?
13. Trace the shapes in the lecture's Keras CNN for a 1×28×28 input, and explain why a second-layer filter has 225 parameters.
14. Why does a CNN fit Go and speech, and how are the filters applied to a spectrogram?
15. What problem does the residual (skip-connection) trick address, and how?

---

## Answer Key

1. **The "Winter of Neural Networks" was caused by several challenges.** The optimization problem was non-convex, requiring many tricks for tuning hyperparameters like the number of layers and hidden units. Additionally, these networks were hard to analyze theoretically, and large labeled datasets were rare during that period.

2. **CNNs leverage three key properties of images.** The first is Locality, where patterns are smaller than the whole image. The second is Translation Invariance, meaning the same patterns can appear in different regions. The third is that Subsampling the pixels will not change the object, allowing the image to be made smaller.

3. **A filter in a convolutional layer is a matrix of parameters (weights) designed to detect a small, specific pattern in the input image, such as an edge or a curve.** This directly relates to the property of Locality, as the filter does not need to see the whole image to discover its target pattern.

4. **Translation invariance is the principle that an object's appearance is independent of its location within an image.** A CNN achieves this through weight sharing, where the same set of filter parameters is applied across all regions of the image to detect a specific pattern regardless of where it appears.

5. **The purpose of the Max Pooling operation is to subsample the feature map created by the convolutional layer.** It reduces the spatial dimensions of the data, which makes the image smaller and results in fewer parameters for the network to process in subsequent layers.

6. **A convolutional layer uses significantly fewer parameters than a fully-connected layer.** This is because its neurons are only connected to a small local region of the input (locality) and it reuses the same parameters across the entire image (weight sharing), whereas every neuron in a fully-connected layer connects to every input from the previous layer.

7. **The lecture reviews two pipelines: (1) split the data into train / validation / test, and (2) run k-fold cross-validation on the training set to choose hyperparameters, then evaluate on the test set.** Model selection uses validation data (or k-CV) to choose the best model; model assessment estimates the chosen model's prediction error on new data.

8. **The Convolutional Neural Network (CNN) was invented by Professor Yann LeCun.** The foundational paper, "Gradient-based learning applied to document recognition," was published in 1998.

9. **After the final max pooling layer, the resulting multi-channel, smaller image (a 3-D tensor) is processed by a Flatten operation.** This converts the 3-D data into a one-dimensional vector, which can then be used as input for the subsequent fully connected feedforward network.

10. **TP and TN count correctly predicted positive and negative samples; FP counts negatives predicted positive, and FN counts positives predicted negative.** Precision = TP / (TP + FP) measures how accurate the positive predictions are, and recall (sensitivity) = TP / (TP + FN) measures coverage of the actual positives. Accuracy = (TP + TN) / (TP + TN + FP + FN), specificity = TN / (TN + FP), and F1 = 2TP / (2TP + FP + FN).

11. **After shifting the image one pixel right, max pooling keeps 2 of the 4 pooled values unchanged, while average pooling keeps 0 of 4.** Max reports the strongest response in each group, so a strong match survives a small shift; average reports the overall response, so values can cancel out. A shift that stays inside a pooling group leaves the max unchanged, but a bigger shift can change it.

12. **A color image is a tensor with three channels (R, G, B) instead of a single matrix.** Each filter therefore also has three channels, one 3×3 slice per color channel, and is applied across all of them.

13. **The input 1×28×28 becomes 25×26×26 after convolution with 25 3×3 filters, 25×13×13 after max pooling, 50×11×11 after a second convolution with 50 filters, and 50×5×5 after pooling; flattening gives 1250 values for the fully connected network.** A first-layer filter has 3×3 = 9 parameters because the input has one channel. A second-layer filter sees all 25 input channels, so it has 25 × 9 = 225 parameters.

14. **A Go board is a 19×19 grid (black 1, white -1, empty 0) that can be treated as an image whose output is the next move among 19×19 positions; a fully connected network can be used, but a CNN performs much better.** For speech, the spectrogram (time × frequency) is the image, and the filters move in the frequency direction.

15. **A deeper network should not have higher training error than a shallower one, yet in practice solvers fail to find good solutions as networks get deeper.** By construction, copying a shallower model's layers and setting the extra layers to the identity gives a deeper model at least as good; residual/skip connections build this idea into the network. ResNet (He et al., CVPR 2016) uses such connections, e.g., the 34-layer residual network compared with VGG-19 and a 34-layer plain network.

---

## Essay Questions

Develop a detailed, essay-format response for each of the following prompts. No answers are provided for this section.

1. **Trace the historical evolution of artificial intelligence and machine learning from Alan Turing's 1950 paper to the landmark 2012 paper on ImageNet classification.** Discuss the key milestones, influential figures, dominant technologies of each era (e.g., expert systems, SVMs), and the factors that contributed to the "Winter of Neural Networks."

2. **Provide a comprehensive explanation of the complete architecture of a standard CNN for image classification, as described in the lecture.** Detail the journey of an input image through the network, explaining the purpose and mechanics of the Convolution layer, Max Pooling layer, Flatten operation, and the final Fully Connected network with softmax.

3. **Elaborate on the three fundamental properties of images that make CNNs more effective than traditional Multilayer Perceptrons (MLPs) for image-related tasks.** For each property (Locality, Translation Invariance, and Subsampling), explain what it is and which specific CNN mechanism (e.g., filters, weight sharing, pooling) is designed to exploit it.

4. **Compare and contrast the processes of "Model Selection" and "Model Assessment."** Describe the two pipelines reviewed in the lecture: (1) a train / validation / test split, and (2) k-fold cross-validation on the training set to choose hyperparameters, followed by evaluation on the test set.

5. **Using the provided example of a 6x6 image and a 3x3 filter, explain the mathematical process of convolution.** Describe how applying the filter across the image generates a "feature map" and how parameters like stride can alter the dimensions of this output.

---

## Glossary of Key Terms

| Term | Definition |
|------|------------|
| **Convolution** | An operation where a filter (a small matrix of parameters) is applied across an input image to produce a feature map. This process leverages the properties of locality and translation invariance. |
| **Convolutional Neural Networks (CNN)** | A type of neural network invented by Yann LeCun, first successfully trained with many layers in 1998. It is specifically designed for data with grid-like topologies, such as 2D images, by leveraging properties of locality, translation invariance, and subsampling. |
| **Confusion Matrix** | A table of the counts of true positives (TP), true negatives (TN), false positives (FP), and false negatives (FN), from which accuracy, precision, recall, specificity, and F1 are computed. |
| **Cross-Validation (k-CV)** | k-fold cross-validation run on the training set to choose hyperparameters, before a final evaluation on the test set. |
| **Feature Map** | The output of applying a filter across an image in a convolutional layer. Each feature map corresponds to a specific filter and represents the detection of that filter's pattern across the input. |
| **EfficientNet** | A family of CNNs (B0 to B7) that reaches higher ImageNet top-1 accuracy with far fewer parameters than earlier models such as ResNet, NASNet, and AmoebaNet. |
| **Filter** | A small matrix of learnable parameters (weights) used in a convolution layer to detect specific patterns (e.g., edges, textures) in an input image. |
| **Flatten** | An operation in a CNN that converts the multi-dimensional output of the convolutional/pooling layers into a single one-dimensional vector. This vector is then fed into a fully connected network. |
| **Fully Connected Feedforward Network** | The final part of a typical CNN architecture. It takes the flattened vector as input and performs classification, often ending with a softmax layer to output probabilities for each class. |
| **Locality** | A property of images where patterns of interest (like a bird's beak) are much smaller than the whole image. CNNs exploit this by using small filters, meaning a neuron does not have to see the entire image to discover a pattern. |
| **Max Pooling** | A subsampling technique where a feature map is downsized by taking the maximum value over a defined window. This reduces the number of parameters and computational complexity, making the representation more robust to small shifts. |
| **Average Pooling** | Subsampling that takes the mean over each window; values can cancel out, so it is less robust to small shifts than max pooling. |
| **Model Assessment** | The process of, having chosen a final model, estimating its prediction error on new, unseen data. |
| **Model Selection** | The process of estimating the performance of different models (or models with different hyperparameters) in order to choose the best one. This is also referred to as hyperparameter tuning. |
| **Residual (Skip) Connection** | A shortcut path that carries a block's input past its layers, as in ResNet. Motivation: a deeper model built from a shallower one plus identity layers should not have higher training error, yet plain deep networks are hard to optimize. |
| **Subsampling** | The property that reducing the resolution of an image (e.g., by taking every other pixel) will not fundamentally change the object depicted. This is the principle behind pooling layers in a CNN. |
| **Translation Invariance** | A property of images where the appearance of an object is independent of its location. CNNs model this by sharing weights, meaning the same filter is used to detect a pattern regardless of where it appears in the image. |
| **Weight Sharing** | The practice in a CNN where the same set of filter parameters is applied across all spatial locations in an input image. This drastically reduces the total number of parameters and enforces translation invariance. |
