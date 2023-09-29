"""
Created on 29/09/2023
@author jdh

The network that takes us from 62x62 to 11 parameters.
"""

from flax import linen as nn  # Linen API


# class CNN(nn.Module):
#     """A simple CNN model."""
#
#     @nn.compact
#     def __call__(self, x):
#         x = nn.Conv(features=32, kernel_size=(3, 3))(x)
#         x = nn.relu(x)
#         x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
#         x = nn.Conv(features=64, kernel_size=(3, 3))(x)
#         x = nn.relu(x)
#         x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
#         x = x.reshape((x.shape[0], -1))  # flatten
#         x = nn.Dense(features=256)(x)
#         x = nn.relu(x)
#         x = nn.Dense(features=11)(x)
#         return x
#

#
# class CNN(nn.Module):
#     num_classes = 12  # Number of output dimensions
#
#     @nn.compact
#     def __call__(self, x):
#         x = nn.Conv(features=32, kernel_size=(3, 3))(x)
#         x = nn.relu(x)
#         x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
#
#         x = nn.Conv(features=64, kernel_size=(3, 3))(x)
#         x = nn.relu(x)
#         x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
#
#         x = nn.Conv(features=128, kernel_size=(3, 3))(x)
#         x = nn.relu(x)
#         x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
#
#         x = x.reshape((x.shape[0], -1))  # Flatten
#
#         x = nn.Dense(features=256)(x)
#         x = nn.relu(x)
#
#         x = nn.Dense(features=self.num_classes)(x)  # Output layer
#         return x


from flax import linen as nn


class CNN(nn.Module):
    num_parameters = 12  # Number of output parameters

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

        # Use a Dense layer with num_parameters units for regression
        parameters = nn.Dense(features=self.num_parameters)(x)
        return parameters


import jax
import jax.numpy as jnp
from flax import linen as nn


class CNN(nn.Module):
    num_parameters = 11  # Number of output parameters

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=32, kernel_size=(3, 3), padding='VALID')(x)  # Specify kernel_size and padding
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))

        x = nn.Conv(features=64, kernel_size=(3, 3), padding='VALID')(x)  # Specify kernel_size and padding
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))

        x = nn.Conv(features=128, kernel_size=(3, 3), padding='VALID')(x)  # Specify kernel_size and padding
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))

        x = x.reshape((x.shape[0], -1))  # Flatten

        x = nn.Dense(features=256)(x)
        x = nn.relu(x)

        # Use a Dense layer with num_parameters units for regression
        parameters = nn.Dense(features=self.num_parameters)(x)
        return parameters
