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

import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from tqdm import tqdm
SEED = 1

import time
# neural network


import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
from jax.scipy.special import logsumexp
from fine_tuning.models.capacitance import do2d


from tensorflow.data import Dataset

def random_layer_params(m, n, key, scale=1e-2):
    w_key, b_key = random.split(key)
    return scale * random.normal(w_key, (n, m)), scale * random.normal(b_key, (n,))


def init_network_params(sizes, key):
    keys = random.split(key, len(sizes))
    return [random_layer_params(m, n, k) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]


layer_sizes = [(62 * 62), (30 * 30), 512, 512, 12]

step_size = 1e-7
num_epochs = 10
batch_size = 128

training_batches = 40
test_batches = 10

shuffle_buffer_size = 100
n_targets = 12
params = init_network_params(layer_sizes, random.PRNGKey(SEED))

number_of_samples = training_batches * batch_size
number_of_test_samples = test_batches * batch_size


def relu(x):
    return jnp.maximum(0, x)

def normalise(data):

    return (data - data.min()) / (data.max() - data.min())

def predict(params, image):
    activations = image
    for w, b in params[:-1]:
        outputs = jnp.dot(w, activations) + b
        activations = relu(outputs)

    final_w, final_b = params[-1]
    logits = jnp.dot(final_w, activations) + final_b

    return logits - logsumexp(logits)


batched_predict = vmap(predict, in_axes=(None, 0))

random_flattened_images = random.normal(random.PRNGKey(SEED), (100, 62 * 62,))
preds = batched_predict(params, random_flattened_images)


cdd_diag_ratio = 8
c_dg_0 = 1
c_dg_1 = 0.05
c_dg_2 = 0.58
c_dg_3 = 1
x_shift = 0.2
y_shift = -0.4
contrast_0 = 1.2
contrast_1 = 1.3
offset = 1
gamma = 1
x0 = 1

initial_params = jnp.array([
    cdd_diag_ratio, c_dg_0, c_dg_1, c_dg_2, c_dg_3,
    x_shift, y_shift, contrast_0, contrast_1, offset,
    gamma, x0
])

params_max = jnp.array([
    10, 2, 0.4, 0.4, 2, 1, 1, 2, 2, 3, 3, 3
])

params_min = jnp.array([
    2, 0.5, 0, 0, 0.5, -1, -1, 0.3, 0.3, 1, 1, 1
])

randomiser = 1 / 2 * random.uniform(random.PRNGKey(SEED), (number_of_samples, params_max.size))
random_params = params_min[jnp.newaxis, :] + randomiser * (params_max - params_min)[jnp.newaxis, :]

test_random = 1 / 2 * random.uniform(random.PRNGKey(SEED), (number_of_test_samples, params_max.size))
test_params = params_min[jnp.newaxis, :] + test_random * (params_max - params_min)[jnp.newaxis, :]

def do2d_(params):
    return do2d(*params)


# def loss(network_params, cc_params):
#     images = do2d(*cc_params).flatten()
#     prediction = batched_predict(network_params, images)
#
#     # sum of squares error over output params
#     return jnp.sum((cc_params - prediction) ** 2)

def accuracy(params, images, targets):
    outputs = batched_predict(params, images)

    return jnp.sum((targets - outputs) ** 2)


def loss(network_params, images, targets):
    preds = batched_predict(network_params, images)
    return jnp.sum((targets - preds) ** 2)

def update(network_params, images, cc_params):
    grads = grad(loss)(network_params, images, cc_params)
    return [(w - step_size * dw, b - step_size * db)
            for (w, b), (dw, db) in zip(params, grads)]



t = time.time()
train_images = vmap(do2d_, in_axes=(0))(random_params).reshape(number_of_samples, -1)
t1 = time.time()
train_labels = random_params

print(f'took {t1 - t} seconds for {number_of_samples}, \n{(t1 - t) / number_of_samples} seconds each ')


test_images = vmap(do2d_, in_axes=(0))(test_params).reshape(number_of_test_samples, -1)
test_labels = test_params

training_set = Dataset.from_tensor_slices((train_images, train_labels))
test_set = Dataset.from_tensor_slices((test_images, test_labels))


training_set = training_set.shuffle(shuffle_buffer_size).batch(batch_size)
test_set = test_set.batch(batch_size)

test_error = []
for epoch in range(num_epochs):
    start_time = time.time()
    for x, y in training_set:
        x = jnp.array(x, dtype=jnp.float32)
        x = normalise(x)
        y = jnp.array(y, dtype=jnp.float32)
        params = update(params, train_images, train_labels)
    epoch_time = time.time() - start_time

    print(f'epoch {epoch} in {epoch_time:0.2f} s')
    print(f'training accuracy: {accuracy(params, train_images, train_labels)}')
    error = accuracy(params, test_images, test_labels)
    test_error.append(error)
    print(f'test error: {error}')



plt.figure()
plt.plot(test_error)
plt.show()