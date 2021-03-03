# mnist-explanation

In this repository, I try to implement explanation methods as described by G. Montavon et al. in their paper `Methods for Interpreting and Understanding Deep Neural Networks` [1]

## Neural Network Implementation

In their original paper, the researchers give advises on the kind of Deep Neural Network (DNN) to be used in order to maximise their explainability. We will follow the following advices:
 - Use as few fully-connected layers as needed to be accurate, and train these layers with dropout
 - Use sum-pooling layers abundantly, and prefer them to other types of pooling layers
 - In the linear layers (convolution and fully-connected), constrain biases to be zero or negative

As such, I am using a very simple CNN: 2 convolutional layers with associated average-pooling layers followed by 3 fully-connected layers, using the Leaky RELU activation function. I use the Adam optimiser (using PyTorch default values) together with the Categorical Cross-Entropy loss function, over 10 epochs.

With 98.89% accuracy over the test set, the result is judged satisfactory enough to be used in the explanation framework.

## Explanation of Neural Network results

Explanation is defined by the researchers as follows: `An explanation is the collection of features of the interpretable domain, that have contributed for a given example to produce a decision (e.g. classification or regression)`.

In this repository, I will try to reproduce some of their methods on the DNN defined above.

### Sensitivity Analysis

In the scope of sensibility analysis, we study the gradient of the output with regard to the input. Here are the examples with regard to the selected samples. They are very similar to the one proposed by the research paper, I thus consider the replication to be successful.

![Results of sensitivity analysis on my test DNN](/pic/sensitivity_analysis.png)

### Simple Taylor Decomposition

The simple Taylor decomposition uses the Taylor series of the gradient to derive a relevance score. In contrary to sensitivity analysis, this score can be either positive or negative and takes into account not only the gradient, but also the input value.

The results of this replication qualitatively match the paper's illustration.

![Results of simple Taylor decomposition on my test DNN](/pic/simple_taylor.png)

### Layer-wise Relevance Propagation

Layer-wise relevance propagation is a conserving backward propagation technique, designed precisely for explanation of DNNs. It is described in more details in another paper by the same author [2]. My implementation borrows a big part of the code proposed by the author as an example on http://heatmapping.org/tutorial

The results of this replication qualitatively match the paper's illustration.

![Results of LRP on my test DNN](/pic/lrp.png)

## References

[1] Grégoire Montavon, Wojciech Samek, Klaus-Robert Müller, Methods for interpreting and understanding deep neural networks, Digital Signal Processing, Volume 73, 2018, Pages 1-15, ISSN 1051-2004, https://doi.org/10.1016/j.dsp.2017.10.011. (https://www.sciencedirect.com/science/article/pii/S1051200417302385)

[2] Montavon G., Binder A., Lapuschkin S., Samek W., Müller KR. (2019) Layer-Wise Relevance Propagation: An Overview. In: Samek W., Montavon G., Vedaldi A., Hansen L., Müller KR. (eds) Explainable AI: Interpreting, Explaining and Visualizing Deep Learning. Lecture Notes in Computer Science, vol 11700. Springer, Cham. https://doi.org/10.1007/978-3-030-28954-6_10
