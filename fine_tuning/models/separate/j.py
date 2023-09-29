"""
Created on 29/09/2023
@author jdh
"""

import jax

import jax
import jax.numpy as jnp
from flax import linen as nn


class CNN(nn.Module):
    num_classes = 11  # Number of output dimensions

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=32, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))

        x = nn.Conv(features=64, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))

        x = nn.Conv(features=128, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))

        x = x.reshape((x.shape[0], -1))  # Flatten

        x = nn.Dense(features=256)(x)
        x = nn.relu(x)

        x = nn.Dense(features=self.num_classes)(x)  # Output layer
        return x


rng = jax.random.PRNGKey(0)
input_shape = (1, 62, 62, 1)

model = CNN()

params = model.init(rng, jnp.ones(input_shape))
example_input = jnp.ones(input_shape)

predictions = model.apply(params, example_input)