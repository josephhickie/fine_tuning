"""
Created on 28/09/2023
@author jdh

recovering the capacitance matrix parameters from the constant capacitance data

1. neural network that takes a 62x62 constant capacitance simulation (for now) as input
    and has an 11-dimensional final layer. These are the same as the inputs for the model.

2. data: i need to generate a load of 62x62 data and include with it the input vectors.

3. training: i will then train the network to go from the 62x62 data to the 11 vector by training to minimise
    the difference between this layer and the true values.

"""

# neural network

import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
from jax.scipy.special import logsumexp


def random_layer_params(m, n, key, scale=1e-2):
    w_key, b_key = random.split(key)
    return scale * random.normal(w_key, (n, m)), scale * random.normal(b_key, (n,))


def init_network_params(sizes, key):
    keys = random.split(key, len(sizes))
    return [random_layer_params(m, n, k) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]


layer_sizes = [(62 * 62), (30 * 30), 512, 512, 11]

step_size = 0.01
num_epochs = 10
batch_size = 128
n_targets = 11
params = init_network_params(layer_sizes, random.PRNGKey(0))


def relu(x):
    return jnp.maximum(0, x)


def predict(params, image):

    activations = image
    for w, b in params[:-1]:
        outputs = jnp.dot(w, activations) + b
        activations = relu(outputs)

    final_w, final_b = params[-1]
    logits = jnp.dot(final_w, activations) + final_b

    return logits - logsumexp(logits)

batched_predict = vmap(predict, in_axes=(None, 0))

random_flattened_images = random.normal(random.PRNGKey(1), (100, 62 * 62, ))
preds = batched_predict(params, random_flattened_images)

print(preds.shape)




