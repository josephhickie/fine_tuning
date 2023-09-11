import jax.numpy as np
import jax
from jax import random, jit, grad
from jax.example_libraries import optimizers

from jax.example_libraries import stax
from jax.example_libraries.stax import Dense, Relu, LogSoftmax, Sigmoid, FanOut, Softplus
from jax.scipy.stats import norm
from functools import partial

class Classifier:
    """
    A simple classifier with two hidden layers and a softmax output layer.
    """
    def __init__(self, d_latent: int=10, d_hidden: int=512):
        """

        :param d_latent:
        :param d_hidden:
        """
        self.d_latent = d_latent
        self.d_hidden = d_hidden
        self._init_network_()

    def _init_network_(self):
        self.classifier_init, self.predict = stax.serial(
        Dense(self.d_hidden), Relu,
        Dense(self.d_hidden), Relu,
        Dense(self.d_latent), LogSoftmax)


class TrainClassifier:
    """
    A class to train the classifier.
    """
    def __init__(self, encode, classifier, step_size, momentum_mass):
        """

        :param encode:
        :param classifier:
        :param step_size:
        :param momentum_mass:
        """
        self.encode = encode
        self.classifier = classifier
        self.step_size = step_size
        self.momentum_mass = momentum_mass
        self._init_optimiser_()

    def _init_optimiser_(self):
        self.opt_init, self.opt_update, self.get_params = optimizers.momentum(self.step_size, mass=self.momentum_mass)

    @partial(jit, static_argnums=(0,))
    def loss(self, classifier_params, batch):
        inputs, targets = batch
        preds = self.classify(inputs, classifier_params)
        return -np.mean(np.sum(preds * targets, axis=1))

    @partial(jit, static_argnums=(0,))
    def accuracy(self, classifier_params, batch):
        inputs, targets = batch
        preds = self.classify(inputs, classifier_params)
        target_class = np.argmax(targets, axis=1)
        predicted_class = np.argmax(preds, axis=1)
        return np.mean(predicted_class == target_class)

    @partial(jit, static_argnums=(0,))
    def classify(self, X, classifier_params):
        z = self.encode(X)
        return self.classifier.predict(classifier_params, z)

    @partial(jit, static_argnums=(0,))
    def update(self, i, opt_state, batch):
        params = self.get_params(opt_state)
        loss_ = lambda params: self.loss(params, batch)
        return self.opt_update(i, grad(loss_)(params), opt_state)