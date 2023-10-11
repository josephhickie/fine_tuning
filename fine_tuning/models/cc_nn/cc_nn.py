"""
Created on 10/10/2023
@author jdh

A neural network that models the constant capacitance model
"""

import matplotlib

matplotlib.use('TkAgg')
from fine_tuning.models import do2d


from tqdm import tqdm

from fine_tuning.models.separate import generate_data
import jax
import jax.numpy as jnp
from jax import random, grad, jit
import flax
import flax.linen as nn
import optax

def do2d_(params):
    return do2d(*params)


def get_data():
    # Load your dataset and preprocess it as needed
    # X_train, y_train, X_test, y_test = load_and_preprocess_data()

    params_max = jnp.array([
        4, 2, 0.1, 0.1, 2, 0.2, 0.2, 2, 2, 3, 5, 10
    ])

    params_min = jnp.array([
        1.5, 0.5, 0.01, 0.01, 0.5, -0.2, -0.2, -2, -2, 1, 1, -10
    ])

    cdd_diag_ratio = 8
    c_dg_0 = 1
    c_dg_1 = 0.05
    c_dg_2 = 0.58
    c_dg_3 = 1
    x_shift = 0.7
    y_shift = 0.2
    contrast_0 = -0.8
    contrast_1 = -1.3
    offset = 0
    gamma = 2
    x0 = -9

    initial_params_ = jnp.array([
        cdd_diag_ratio, c_dg_0, c_dg_1, c_dg_2, c_dg_3,
        x_shift, y_shift, contrast_0, contrast_1, offset,
        gamma, x0
    ])

    n_training_samples = 20000
    n_test_samples = 10000
    (y_train, X_train), (y_test, X_test) = generate_data(n_training_samples, n_test_samples, params_min, params_max)

    return (y_train, X_train), (y_test, X_test)


# Define your custom neural network model
class NN(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(512)(x)
        x = nn.relu(x)
        x = nn.Dense(2048)(x)
        x = nn.relu(x)
        x = nn.Dense(62 * 62)(x)
        return x



# Main function for training and evaluation
# def main():
# Load your dataset and preprocess it
(y_train, X_train), (y_test, X_test) = get_data()

# Define input and output dimensions based on your data
input_dim = 12
output_dim = 62 * 62  # 62x62 pixels

# Create a random key for initializing the model parameters
key = random.PRNGKey(0)

# Initialize the model
model = NN()
initial_params = model.init(key, jnp.ones((1, input_dim)))


# Initialize the optimizer
opt = optax.adam(learning_rate=0.0007)
opt_state = opt.init(initial_params)

# Define batch size and number of training epochs
batch_size = 64
num_epochs = 200

# Define a function to compute the mean squared error loss
def mse_loss(params, inputs, targets):
    outputs = model.apply(params, inputs)
    return jnp.mean((outputs - targets) ** 2)

# Define a training step function
@jit
def train_step(params, opt_state, batch):
    inputs, targets = batch
    gradients = grad(mse_loss)(params, inputs, targets)
    updates, new_opt_state = opt.update(gradients, opt_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state


# Training loop
for epoch in range(num_epochs):
    key, subkey = random.split(key)
    indices = jax.random.permutation(subkey, len(X_train))
    for batch_start in range(0, len(X_train), batch_size):
        batch_indices = indices[batch_start:batch_start + batch_size]
        batch_inputs = X_train[batch_indices]
        batch_targets = y_train[batch_indices]
        initial_params, opt_state = train_step(initial_params, opt_state, (batch_inputs, batch_targets))

    # Calculate and print the training loss for this epoch
    train_loss = mse_loss(initial_params, X_train, y_train)
    print(f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {train_loss}")

# Evaluation
test_loss = mse_loss(initial_params, X_test, y_test)
print(f"Test Loss: {test_loss}")

