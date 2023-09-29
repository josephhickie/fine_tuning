"""
Created on 29/09/2023
@author jdh
"""

from datasets import generate_data

from jax.example_libraries import stax, optimizers
from jax import numpy as jnp
import jax

print("JAX Version : {}".format(jax.__version__))

from tensorflow import keras
from sklearn.model_selection import train_test_split

n_training_samples = 20000
n_test_samples = 10000
params_max = jnp.array([
    10, 2, 0.4, 0.4, 2, 1, 1, 2, 2, 3, 3, 3
])

params_min = jnp.array([
    2, 0.5, 0, 0, 0.5, -1, -1, 0.3, 0.3, 1, 1, 1
])


def normalise(data):
    return (data - data.min()) / (data.max() - data.min())


(X_train, Y_train), (X_test, Y_test) = generate_data(n_training_samples, n_test_samples, params_min, params_max)
# (X_train, Y_train), (X_test, Y_test) = keras.datasets.fashion_mnist.load_data()

X_train, X_test, Y_train, Y_test = jnp.array(X_train, dtype=jnp.float32), \
    jnp.array(X_test, dtype=jnp.float32), \
    jnp.array(Y_train, dtype=jnp.float32), \
    jnp.array(Y_test, dtype=jnp.float32)

X_train, X_test = X_train.reshape(-1, 62, 62, 1), X_test.reshape(-1, 62, 62, 1)

X_train, X_test = normalise(X_train), normalise(X_test)

classes = range(12)

X_train.shape, X_test.shape, Y_train.shape, Y_test.shape

conv_init, conv_apply = stax.serial(
    stax.Conv(32, (3, 3), padding="SAME"),
    stax.Relu,
    stax.Conv(16, (3, 3), padding="SAME"),
    stax.Relu,

    stax.Flatten,
    stax.Dense(len(classes)),
    stax.Softmax
)

rng = jax.random.PRNGKey(123)

weights = conv_init(rng, (1, 62, 62, 1))

weights = weights[1]  ## Weights are actually stored in second element of two value tuple

for w in weights:
    if w:
        w, b = w
        print("Weights : {}, Biases : {}".format(w.shape, b.shape))

preds = conv_apply(weights, X_train[:5])


def CrossEntropyLoss(weights, input_data, actual):
    preds = conv_apply(weights, input_data)
    return jnp.sum((actual - preds)**2)
    # one_hot_actual = jax.nn.one_hot(actual, num_classes=len(classes))
    # log_preds = jnp.log(preds)
    # return - jnp.sum(one_hot_actual * log_preds)


from jax import value_and_grad


def TrainModelInBatches(X, Y, epochs, opt_state, batch_size=32):
    for i in range(1, epochs + 1):
        batches = jnp.arange((X.shape[0] // batch_size) + 1)  ### Batch Indices

        losses = []  ## Record loss of each batch
        for batch in batches:
            if batch != batches[-1]:
                start, end = int(batch * batch_size), int(batch * batch_size + batch_size)
            else:
                start, end = int(batch * batch_size), None

            X_batch, Y_batch = X[start:end], Y[start:end]  ## Single batch of data

            loss, gradients = value_and_grad(CrossEntropyLoss)(opt_get_weights(opt_state), X_batch, Y_batch)

            ## Update Weights
            opt_state = opt_update(i, gradients, opt_state)

            losses.append(loss)  ## Record Loss

        print("CrossEntropyLoss : {:.3f}".format(jnp.array(losses).mean()))

    return opt_state


seed = jax.random.PRNGKey(123)
learning_rate = jnp.array(1 / 1e4)
epochs = 25
batch_size = 256

weights = conv_init(rng, (batch_size, 28, 28, 1))
weights = weights[1]

opt_init, opt_update, opt_get_weights = optimizers.sgd(learning_rate)
opt_state = opt_init(weights)

final_opt_state = TrainModelInBatches(X_train, Y_train, epochs, opt_state, batch_size=batch_size)


def MakePredictions(weights, input_data, batch_size=32):
    batches = jnp.arange((input_data.shape[0] // batch_size) + 1)  ### Batch Indices

    preds = []
    for batch in batches:
        if batch != batches[-1]:
            start, end = int(batch * batch_size), int(batch * batch_size + batch_size)
        else:
            start, end = int(batch * batch_size), None

        X_batch = input_data[start:end]

        if X_batch.shape[0] != 0:
            preds.append(conv_apply(weights, X_batch))

    return preds


test_preds = MakePredictions(opt_get_weights(final_opt_state), X_test, batch_size=batch_size)

test_preds = jnp.concatenate(test_preds).squeeze()  ## Combine predictions of all batches

test_preds = jnp.argmax(test_preds, axis=1)

train_preds = MakePredictions(opt_get_weights(final_opt_state), X_train, batch_size=batch_size)

train_preds = jnp.concatenate(train_preds).squeeze()  ## Combine predictions of all batches

train_preds = jnp.argmax(train_preds, axis=1)

test_preds[:5], train_preds[:5]

from sklearn.metrics import accuracy_score

print("Train Accuracy : {:.3f}".format(accuracy_score(Y_train, train_preds)))
print("Test  Accuracy : {:.3f}".format(accuracy_score(Y_test, test_preds)))
