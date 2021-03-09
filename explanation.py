### CODE PARTIALLY TAKEN FROM: http://www.heatmapping.org/tutorial ###

import numpy as np
import torch
import torch.nn as nn
from torchvision.transforms.functional import affine
import copy

class Explanation:
    def __init__(self, model):
        assert isinstance(model, nn.Module), f'Explanation framework expects model to be {nn.Module}'
        self.model = model

    def __call__(self, X, y_true):
        return NotImplementedError('Method needs to be overwritten by subclass')

class SensitivityAnalysis(Explanation):
    def __init__(self, model, loss):
        super().__init__(model)
        assert isinstance(loss, nn.Module), f'Explanation framework expects loss to be {nn.Module}'
        self.loss = loss

    def __call__(self, X, y_true):
        """Computes the sensitivity analysis for the given inputs and label

        Args:
            X (torch.Tensor): 4-dimension Tensor representing the input to be explained
            y_true (torch.Tensor): 2-dimension truth labels corresponding to given input

        Returns:
            torch.Tensor: Relevance heatmaps
            torch.Tensor: Predicted labels
        """

        X.requires_grad = True
        y_pred = self.model(X)
        loss = self.loss(y_pred, y_true)
        loss.backward(retain_graph=True)
        return X.grad ** 2, y_pred

class SimpleTaylor(Explanation):
    def __init__(self, model, loss):
        super().__init__(model)
        assert isinstance(loss, nn.Module), f'Explanation framework expects loss to be {nn.Module}'
        self.loss = loss

    def __call__(self, X, y_true):
        """Computes the simple taylor decomposition for the given inputs and label

        Args:
            X (torch.Tensor): 4-dimension Tensor representing the input to be explained
            y_true (torch.Tensor): 2-dimension truth labels corresponding to given input

        Returns:
            torch.Tensor: Relevance heatmaps
            torch.Tensor: Predicted labels
        """

        X.requires_grad = True
        y_pred = self.model(X)
        loss = self.loss(y_pred, y_true)
        loss.backward(retain_graph=True)
        return X.grad * X, y_pred

class LRP(Explanation):
    def __init__(self, model, mean, std, denoising=0):
        """Initialises LRP object

        Args:
            model (torch.nn.Module): Model to be explained
            mean: Global mean of dataset
            std: Global standard deviation of dataset
            denoising: Translation pixels in every direction, for de-noising purposes
        """
        super().__init__(model)
        self.mean = mean
        self.std = std
        self.denoising = denoising

    def __call__(self, X, y_true):
        """Computes the LRP for the given inputs and label

        Args:
            X (torch.Tensor): 4-dimension Tensor representing the input to be explained
            y_true (torch.Tensor): 2-dimension truth labels corresponding to given input

        Returns:
            torch.Tensor: Relevance heatmaps
            torch.Tensor: Predicted labels
        """
        translations = 1 + self.denoising * 2
        heatmaps = torch.empty([translations ** 2] + list(X.shape))

        for i in range(translations):
            for j in range(translations):
                X_tr = affine(X,
                              translate=[i - self.denoising, j - self.denoising],
                              angle=0., scale=1., shear=[0., 0.])
                R = self.__lrp(X_tr, y_true)
                heatmaps[i * translations + j] = affine(R,
                                                        translate=[-i + self.denoising, -j + self.denoising],
                                                        angle=0., scale=1., shear=[0., 0.])

        return torch.mean(heatmaps, axis=0), self.model(X)

    def __lrp(self, X, y_true):
        # Preparing inputs
        layers = list(self.model._modules.values())
        L = len(layers)

        # Forward pass
        A = [X] + [None] * L
        for l in range(L):
            A[l + 1] = layers[l].forward(A[l])

        # Masking to preserve only true class
        mask = np.zeros((y_true.shape[0], 10))
        for i in range(y_true.shape[0]):
            mask[i][int(y_true[i])] = 1.
        mask = torch.FloatTensor(mask)
        R = [None] * L + [(A[-1] * mask).data]

        # Backward pass
        for l in range(1, L)[::-1]:

            A[l] = (A[l].data).requires_grad_(True)

            if isinstance(layers[l], torch.nn.MaxPool2d): layers[l] = torch.nn.AvgPool2d(2)

            if isinstance(layers[l], nn.Conv2d) \
                    or isinstance(layers[l], torch.nn.AvgPool2d) \
                    or isinstance(layers[l], nn.Linear):

                if l <= 16:       rho = lambda p: p + 0.25 * p.clamp(min=0); incr = lambda z: z + 1e-9
                if 17 <= l <= 30: rho = lambda p: p; incr = lambda z: z + 1e-9 + 0.25 * ((z ** 2).mean() ** .5).data
                if l >= 31:       rho = lambda p: p; incr = lambda z: z + 1e-9

                z = incr(self.__new_layer(layers[l], rho).forward(A[l]))  # step 1
                s = (R[l + 1] / z).data  # step 2
                (z * s).sum().backward();
                c = A[l].grad  # step 3
                R[l] = (A[l] * c).data  # step 4

            elif isinstance(layers[l], nn.Flatten):
                R[l] = R[l + 1].reshape(10, 16, 5, 5)

            else:
                R[l] = R[l + 1]

        # Special case of input layer
        A[0] = (A[0].data).requires_grad_(True)

        lb = (A[0].data * 0 + (0 - self.mean) / self.std).requires_grad_(True)
        hb = (A[0].data * 0 + (1 - self.mean) / self.std).requires_grad_(True)

        z = layers[0].forward(A[0]) + 1e-9  # step 1 (a)
        z -= self.__new_layer(layers[0], lambda p: p.clamp(min=0)).forward(lb)  # step 1 (b)
        z -= self.__new_layer(layers[0], lambda p: p.clamp(max=0)).forward(hb)  # step 1 (c)
        s = (R[1] / z).data  # step 2
        (z * s).sum().backward();
        c, cp, cm = A[0].grad, lb.grad, hb.grad  # step 3
        R[0] = (A[0] * c + lb * cp + hb * cm).data  # step 4

        return R[0]

    @staticmethod
    def __new_layer(layer, g):

        layer = copy.deepcopy(layer)

        try:
            layer.weight = nn.Parameter(g(layer.weight))
        except AttributeError:
            pass

        try:
            layer.bias = nn.Parameter(g(layer.bias))
        except AttributeError:
            pass

        return layer