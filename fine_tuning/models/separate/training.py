"""
Created on 29/09/2023
@author jdh
"""

import optax
import jax.numpy as jnp

from network import CNN
import jax

from datasets import generate_data


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

# Define your loss function for regression, e.g., Mean Squared Error (MSE)
def mse_loss(predictions, targets):
    return jnp.mean((predictions - targets) ** 2)

# Initialize the model, optimizer, and random number generator
model = CNN()
optimiser = optax.adam(learning_rate=1e-3)
rng = jax.random.PRNGKey(0)

# Training function
@jax.jit
def train_step(params, optimiser_state, images, targets):
    def loss_fn(params):
        predictions = model.apply(params, images)
        loss = mse_loss(predictions, targets)
        return loss

    gradient, loss = jax.grad(loss_fn, has_aux=True)(params)
    updates, optimizer_state = optimiser.update(gradient, optimiser_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, optimizer_state, loss

# Initialize model parameters and optimizer state
params = model.init(rng, jnp.ones((batch_size, 1, 62, 62, 1)))
optimiser_state = optimiser.init(params)

# Training loop
for epoch in range(num_epochs):
    # Iterate through your dataset in batches
    for batch_images, batch_targets in train_ds:
        batch_images = jnp.array(batch_images, dtype=jnp.float32)
        batch_targets = jnp.array(batch_targets, dtype=jnp.float32)
        params, optimiser_state, loss = train_step(params, optimiser_state, batch_images, batch_targets)

    print(f"Epoch {epoch + 1}, Loss: {loss:.4f}")

# Now, 'params' contains the trained model parameters

