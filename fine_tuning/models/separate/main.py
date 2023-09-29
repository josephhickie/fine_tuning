"""
Created on 29/09/2023
@author jdh
"""

import tensorflow as tf
import jax
import jax.numpy as jnp  # JAX NumPy

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from network import CNN
from datasets import generate_data, get_datasets
from helpers import create_train_state, compute_metrics, train_step

cnn = CNN()
print(cnn.tabulate(jax.random.PRNGKey(1), jnp.ones((1, 62, 62, 1))))

tf.random.set_seed(0)
init_rng = jax.random.PRNGKey(0)

params_max = jnp.array([
    10, 2, 0.4, 0.4, 2, 1, 1, 2, 2, 3, 3, 3
])

params_min = jnp.array([
    2, 0.5, 0, 0, 0.5, -1, -1, 0.3, 0.3, 1, 1, 1
])

num_epochs = 10
batch_size = 32
n_training_samples = num_epochs * batch_size * 10
n_test_samples = num_epochs * batch_size * 5

train_ds, test_ds = generate_data(n_training_samples, n_test_samples,
                                  params_min, params_max, batch_size=batch_size)

# train_ds, test_ds = get_datasets(num_epochs, batch_size)

learning_rate = 0.01
momentum = 0.9

input_shape = (1, 62, 62, 1)

state = create_train_state(cnn, input_shape, init_rng, learning_rate, momentum)
del init_rng  # Must not be used anymore.

# since train_ds is replicated num_epochs times in get_datasets(), we divide by num_epochs
num_steps_per_epoch = train_ds.cardinality().numpy() // num_epochs

metrics_history = {'train_loss': [],
                   'train_accuracy': [],
                   'test_loss': [],
                   'test_accuracy': []}

# for step, batch in enumerate(train_ds.as_numpy_iterator()):
#
#     # Run optimization steps over training batches and compute batch metrics
#     state = train_step(state, batch)  # get updated train state (which contains the updated parameters)
#     state = compute_metrics(state=state, batch=batch)  # aggregate batch metrics
#
#     if (step + 1) % num_steps_per_epoch == 0:  # one training epoch has passed
#         for metric, value in state.metrics.compute().items():  # compute metrics
#             metrics_history[f'train_{metric}'].append(value)  # record metrics
#         state = state.replace(metrics=state.metrics.empty())  # reset train_metrics for next training epoch
#
#         # Compute metrics on the test set after each training epoch
#         test_state = state
#         for test_batch in test_ds.as_numpy_iterator():
#             test_state = compute_metrics(state=test_state, batch=test_batch)
#
#         for metric, value in test_state.metrics.compute().items():
#             metrics_history[f'test_{metric}'].append(value)
#
#         print(f"train epoch: {(step + 1) // num_steps_per_epoch}, "
#               f"loss: {metrics_history['train_loss'][-1]}, "
#               f"accuracy: {metrics_history['train_accuracy'][-1] * 100}")
#         print(f"test epoch: {(step + 1) // num_steps_per_epoch}, "
#               f"loss: {metrics_history['test_loss'][-1]}, "
#               f"accuracy: {metrics_history['test_accuracy'][-1] * 100}")


@jax.jit
def pred_step(state, batch):
    logits = state.apply_fn({'params': state.params}, test_batch['image'])
    return logits.argmax(axis=1)


test_batch = test_ds.as_numpy_iterator().next()
pred = pred_step(state, test_batch)
fig, axs = plt.subplots(5, 5, figsize=(12, 12))
for i, ax in enumerate(axs.flatten()):
    ax.imshow(test_batch['image'][i, ..., 0], cmap='gray')
    ax.set_title(f"label={pred[i]}")
    ax.axis('off')
