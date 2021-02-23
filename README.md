# mnist-explanation

In this repository, I try to implement explanation methods as described by G. Montavon et al. in their paper `Methods for Interpreting and Understanding Deep Neural Networks` [1]

## Neural Network Implementation

In their original paper, the researchers do not explicit the structure of the Deep Neural Network (DNN) used in the scope of their study, but only mention a composition of convolutional and fully-connected layers.

As such, I am using a very simple CNN: 2 convolutional layers with associated max-pooling layers followed by 3 fully-connected layers, using the RELU activation function. I use the Adam optimiser (using PyTorch default values) together with the Categorical Cross-Entropy loss function, over 10 epochs.

With 98.69% accuracy over the test set, the result is judged satisfactory enough to be used in the explanation framework.


## References

[1] Grégoire Montavon, Wojciech Samek, Klaus-Robert Müller, Methods for interpreting and understanding deep neural networks, Digital Signal Processing, Volume 73, 2018, Pages 1-15, ISSN 1051-2004, https://doi.org/10.1016/j.dsp.2017.10.011. (https://www.sciencedirect.com/science/article/pii/S1051200417302385)
