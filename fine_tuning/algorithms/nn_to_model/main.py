import pickle
import time
import itertools
import numpy.random as npr
from tqdm.auto import trange
import jax.numpy as np
from jax import jit, grad, random, device_put
import matplotlib.pyplot as plt

from vae import VAE, TrainVAE
from classifier import Classifier, TrainClassifier
from utils import data_stream, plot_examples, load_data, load_stability_data

from fine_tuning.models.separate import generate_data

# import orbax.checkpoint
# from flax.training import orbax_utils

from jax.lib import xla_bridge
print(xla_bridge.get_backend().platform)

### ----------------------------------------------------------------------------------------------- ###
### -------------------------------------- VAE Training ------------------------------------------- ###
### ----------------------------------------------------------------------------------------------- ###

# __________________________________________________________________________________________________ #
# __________________________________________________________________________________________________ #
# load the results and the models
# X_train, X_test, y_train, y_test = load_stability_data()
import jax.numpy as jnp


n_training_samples = 20000
n_test_samples = 10000
params_max = jnp.array([
    10, 2, 0.4, 0.4, 2, 1, 1, 2, 2, 3, 3, 3
])

params_min = jnp.array([
    2, 0.5, 0, 0, 0.5, -1, -1, 0.3, 0.3, 1, 1, 1
])


(X_train, X_test), (y_train, y_test) = generate_data(n_training_samples, n_test_samples, params_min, params_max)

d_obs = X_train.shape[1]

# __________________________________________________________________________________________________ #
# __________________________________________________________________________________________________ #

# Define the VAE
d_latent = 12
vae_hidden_layers = 3
vae = VAE(d_obs, n_dense_layers=vae_hidden_layers, d_latent=d_latent)
vae_trainer = TrainVAE(vae, step_size=1e-3)

#__________________________________________________________________________________________________ #
# __________________________________________________________________________________________________ #
# Set up the training loop
enc_init_rng, dec_init_rng = random.split(random.PRNGKey(2))

_, init_encoder_params = vae.encoder_init(enc_init_rng, (-1, d_obs))
_, init_decoder_params = vae.decoder_init(dec_init_rng, (-1, d_latent))
init_params = init_encoder_params, init_decoder_params
vae_opt_state = vae_trainer.opt_init(init_params)
itercount = itertools.count()

print("\nStarting VAE training...")

num_epochs = 5000
batch_size = 128

num_train = X_train.shape[0]
num_complete_batches, leftover = divmod(num_train, batch_size)
num_batches = num_complete_batches + bool(leftover)

batches = data_stream(X_train, y_train, num_train, batch_size, num_batches)

test_rng = random.PRNGKey(0)
pbar = trange(num_epochs)
test_elbos = []
train_elbos = []
for epoch in pbar:
    start_time = time.time()
    epoch_rng = random.PRNGKey(epoch)
    for _ in range(num_batches):
        batch, _ = next(batches)
        vae_opt_state = vae_trainer.update(epoch_rng, next(itercount), vae_opt_state, batch)
    if epoch % 10 == 0:
        elbo_rng, data_rng = random.split(test_rng)
        binarized_test = random.bernoulli(data_rng, X_test)
        test_l = vae_trainer.elbo(elbo_rng, vae_trainer.get_params(vae_opt_state), binarized_test)
        test_elbos.append(test_l)

        binarized_train = random.bernoulli(data_rng, X_train[:2000])
        train_l = vae_trainer.elbo(elbo_rng, vae_trainer.get_params(vae_opt_state), binarized_train)
        train_elbos.append(train_l)


    pbar.set_description("ELBO: {:.1f}".format(test_l))
    epoch_time = time.time() - start_time

plt.plot(train_elbos, '--.', label='Train ELBO')
plt.plot(test_elbos, '--.', label='Test ELBO')
plt.xlabel('Epoch')
plt.ylabel('ELBO')
plt.legend()
plt.title(f'VAE ELBO')
plt.show()

test_elbo=test_l

#__________________________________________________________________________________________________ #
# __________________________________________________________________________________________________ #
# Test trained VAE
vae_params = vae_trainer.get_params(vae_opt_state)

### ----------------------------------------------------------------------------------------------- ###
### ---------------------------------- Classifier Training ---------------------------------------- ###
### ----------------------------------------------------------------------------------------------- ###

# __________________________________________________________________________________________________ #
# __________________________________________________________________________________________________ #
# load the models
d_states = 4
classifier_hidden_layers = 2
classifier = Classifier(d_states=d_states, n_dense_layers = classifier_hidden_layers)

# __________________________________________________________________________________________________ #
# __________________________________________________________________________________________________ #
# Set up the optimiser
step_size = 1e-3
num_epochs = 1000

encode = lambda x: vae.encoder(vae_params[0], x)[0]
classifier_trainer = TrainClassifier(encode, classifier, step_size)

# __________________________________________________________________________________________________ #
# __________________________________________________________________________________________________ #
# Set up the training loop
rng = random.PRNGKey(0)
_, init_classifier_params = classifier.classifier_init(rng, (-1, d_latent))
classifier_opt_state = classifier_trainer.opt_init(init_classifier_params)
itercount = itertools.count()

print("\nStarting classifier training...")
pbar = trange(num_epochs)
test_losses = []
training_losses = []
for epoch in pbar:
    start_time = time.time()
    for _ in range(num_batches):
      classifier_opt_state = classifier_trainer.update(next(itercount), classifier_opt_state, next(batches))
    epoch_time = time.time() - start_time
    if epoch%10==0:
        test_l = classifier_trainer.loss(classifier_trainer.get_params(classifier_opt_state), (X_test[:2000], y_test[:2000]))
        test_losses.append(test_l)
        train_l = classifier_trainer.loss(classifier_trainer.get_params(classifier_opt_state), (X_train[:2000], y_train[:2000]))
        training_losses.append(train_l)
    pbar.set_description("Loss: {:.3f}".format(test_l))

plt.plot(training_losses, '--.', label='Training loss')
plt.plot(test_losses, '--.', label='Test loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title(f'Classifier Loss, classifier dense layers = {classifier_hidden_layers}')
plt.show()

classifier_params = classifier_trainer.get_params(classifier_opt_state)
train_acc = classifier_trainer.accuracy(classifier_params, (X_train, y_train))
test_acc = classifier_trainer.accuracy(classifier_params, (X_test, y_test))
print(f'classifier dense layers = {classifier_hidden_layers}')
print(f"Epoch {epoch} in {epoch_time:0.2f} sec")
print(f"Training set accuracy {train_acc}")
print(f"Test set accuracy {test_acc}")
print('------------------------------------')

def example():
    plot_examples(X_test,
                  lambda x: np.argmax(classifier_trainer.classify(x, classifier_params)),
                  lambda x: vae._reconstruct(x, vae_params),
                  title=f'latent dims = {d_latent}'
                  )

example()
ckpt = {'classifier_optimiser': classifier_opt_state,
        'classifier_params': classifier_params,
        'classifier_dense_layers': classifier_hidden_layers,
        'vae_optimiser': vae_opt_state,
        'vae_params': vae_params,
        'vae_dense_layers':vae_hidden_layers,
        'd_latent': d_latent,
        'd_obs': d_obs}


pth = '/tmp/pycharm_project_790/fine_tuning/fine_tuning/algorithms/saved_models/'
import sys
import pickle
sys.path.append(pth)
file = pth + 'vae_and_classifier.pickle'
# with open(file, 'wb') as f:
#     pickle.dump(ckpt, f)

# def save(filename):
#     orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
#     save_args = orbax_utils.save_args_from_target(ckpt)
#     orbax_checkpointer.save(f'tmp/orbax/{filename}', ckpt, save_args=save_args)
