"""
Created on 12/10/2023
@author jdh
"""

from flax import linen as nn
import jax.numpy as jnp
import jax
import optax


class ResNet(nn.Module):
    num_classes: int

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(64, (7, 7), (2, 2), 'SAME', kernel_init=nn.initializers.xavier_uniform())(x)
        x = nn.BatchNorm(use_running_average=True)(x)
        x = nn.relu(x)

        x = nn.max_pool(x, (3, 3), (2, 2), 'VALID')

        # Use ResNet blocks here
        x = ResNetBlock(64)(x)
        x = ResNetBlock(128)(x)
        x = ResNetBlock(256)(x)
        x = ResNetBlock(512)(x)

        x = jnp.mean(x, axis=(1, 2))
        x = nn.Dense(self.num_classes, kernel_init=nn.initializers.xavier_uniform())(x)

        return x


class ResNetBlock(nn.Module):
    num_channels: int

    @nn.compact
    def __call__(self, x):
        residual = x
        x = nn.Conv(self.num_channels, (3, 3), (1, 1), 'SAME', kernel_init=nn.initializers.xavier_uniform())(x)
        x = nn.BatchNorm(use_running_average=True)(x)
        x = nn.relu(x)
        x = nn.Conv(self.num_channels, (3, 3), (1, 1), 'SAME', kernel_init=nn.initializers.xavier_uniform())(x)
        x = nn.BatchNorm(use_running_average=True)(x)
        x = x + residual
        x = nn.relu(x)
        return x

def train_step(model, batch, optimizer):
    def loss_fn(params):
        logits = model.apply({'params': params}, batch['image'])
        loss = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=batch['label']))
        return loss

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grad = grad_fn(optimizer.target)
    optimizer = optimizer.apply_gradient(grad)
    return optimizer, loss


# Define your optimizer and initialize it
model = ResNet(num_classes=4)
optimizer = optax.adam(learning_rate=1e-3)

params = model.init(rng, )



# Training loop
for epoch in range(num_epochs):
    for batch in training_data:
        optimizer, loss = train_step(model, batch, optimizer)
        # Optionally, log and save the loss

# Save the trained model
model.save('my_image_classifier.pkl')
