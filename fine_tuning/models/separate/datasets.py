"""
Created on 29/09/2023
@author jdh
"""

import jax.numpy as jnp
from jax import random
from jax import vmap
from tensorflow.data import Dataset

from params_to_cc import do2d_

def generate_data(n_training_samples, n_test_samples, params_min, params_max,
                  batch_size=128, SEED=0, shuffle_buffer_size=128
                  ):
    train_random = 1 / 2 * random.uniform(random.PRNGKey(SEED), (n_training_samples, params_max.size))
    train_params = params_min[jnp.newaxis, :] + train_random * (params_max - params_min)[jnp.newaxis, :]
    train_images = vmap(do2d_, in_axes=(0))(train_params).reshape(n_training_samples, -1)

    test_random = 1 / 2 * random.uniform(random.PRNGKey(SEED), (n_test_samples, params_max.size))
    test_params = params_min[jnp.newaxis, :] + test_random * (params_max - params_min)[jnp.newaxis, :]
    test_images = vmap(do2d_, in_axes=(0))(test_params).reshape(n_test_samples, -1)

    # training_set = Dataset.from_tensor_slices((train_images, train_params))
    # test_set = Dataset.from_tensor_slices((test_images, test_params))
    #
    # training_set = training_set.shuffle(shuffle_buffer_size).batch(batch_size)
    # test_set = test_set.batch(batch_size)
    return (train_images, train_params), (test_images, test_params)
    return training_set, test_set

# import tensorflow_datasets as tfds  # TFDS for MNIST
# import tensorflow as tf             # TensorFlow operations
#
# def get_datasets(num_epochs, batch_size):
#   """Load MNIST train and test datasets into memory."""
#   train_ds = tfds.load('mnist', split='train')
#   test_ds = tfds.load('mnist', split='test')
#
#   train_ds = train_ds.map(lambda sample: {'image': tf.cast(sample['image'],
#                                                            tf.float32) / 255.,
#                                           'label': sample['label']}) # normalize train set
#   test_ds = test_ds.map(lambda sample: {'image': tf.cast(sample['image'],
#                                                          tf.float32) / 255.,
#                                         'label': sample['label']}) # normalize test set
#
#   train_ds = train_ds.repeat(num_epochs).shuffle(1024) # create shuffled dataset by allocating a buffer size of 1024 to randomly draw elements from
#   train_ds = train_ds.batch(batch_size, drop_remainder=True).prefetch(1) # group into batches of batch_size and skip incomplete batch, prefetch the next sample to improve latency
#   test_ds = test_ds.shuffle(1024) # create shuffled dataset by allocating a buffer size of 1024 to randomly draw elements from
#   test_ds = test_ds.batch(batch_size, drop_remainder=True).prefetch(1) # group into batches of batch_size and skip incomplete batch, prefetch the next sample to improve latency
#
#   return train_ds, test_ds