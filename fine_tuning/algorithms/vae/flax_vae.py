"""
Created on 08/09/2023
@author jdh
"""

import jax.numpy as np
import jax
from jax import random, jit, grad
from jax.example_libraries import optimizers

from jax.example_libraries import stax
from jax.example_libraries.stax import Dense, Relu, LogSoftmax, Sigmoid, FanOut, Softplus
from jax.scipy.stats import norm
from functools import partial


from flax import linen as nn          # The Linen API
from flax.training import train_state
import optax                          # The Optax gradient processing and optimization library



class ConvEncoder(nn.Module):

    d_obs: int
    d_latent: int
    d_hidden: int

    def _init_encoder_(self):
        pass

    @nn.compact
    def __call__(self, x):

        x = nn.Conv(features=62, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Conv(features=64, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))  # Flatten
        x = nn.Dense(features=self.d_hidden)(x)
        x = nn.relu(x)
        mean_x = nn.Dense(self.d_latent, name='fc2_mean')(x)
        logvar_x = nn.Dense(self.d_latent, name='fc2_logvar')(x)

        return mean_x, logvar_x

class ConvDecoder(nn.Module):

    d_latent: int
    d_obs: int
    d_hidden: int

    @nn.compact
    def __call__(self, x):


        x = nn.Dense(features=self.d_hidden * 2 * 64)(x)
        x = nn.relu(x)
        x = x.reshape(x.shape[0], 8, 8, -1)
        x = nn.ConvTranspose(features=2 * self.d_hidden, kernel_size=(3, 3), strides=(2, 2))(x)
        x = nn.relu(x)
        x = nn.Conv(features=self.d_hidden, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.ConvTranspose(features=1, kernel_size=(3, 3), strides=(2, 2), padding=2)(x)
        x = nn.relu(x)
        x = nn.Conv(features=2*self.d_hidden, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.ConvTranspose(features=1, kernel_size=(2, 2), strides=(2, 2), padding=-1)(x)
        x = nn.tanh(x)
        return x


class ConVAE:

    def __init__(self, d_obs, d_latent):
        self.d_obs = d_obs
        self.d_latent = d_latent
        self.setup()

    def setup(self):
        self._encoder = ConvEncoder(d_obs=self.d_obs, d_latent=self.d_latent, d_hidden=512)
        self._decoder = ConvDecoder(d_obs=self.d_obs, d_latent=self.d_latent, d_hidden=512)

    def encoder(self, params, batch):
        z = self._encoder.apply({'params': params}, batch)
        return z

    def decoder(self, params, batch):
        z = self._decoder.apply({'params': params}, batch)
        return z

    def __call__(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

