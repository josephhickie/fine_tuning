import time
import itertools
import numpy.random as npr
from tqdm.auto import trange
import jax.numpy as np
from jax import jit, grad, random
import matplotlib.pyplot as plt

from vae import VAE, TrainVAE
from classifier import Classifier, TrainClassifier
from utils import data_stream, plot_examples, load_data, load_stability_data

import orbax.checkpoint
from flax.training import orbax_utils



### ----------------------------------------------------------------------------------------------- ###
### -------------------------------------- VAE Training ------------------------------------------- ###
### ----------------------------------------------------------------------------------------------- ###

# __________________________________________________________________________________________________ #
# __________________________________________________________________________________________________ #
# load the data and the models
X_train, X_test, y_train, y_test = load_stability_data()
d_obs = X_train.shape[1]

# __________________________________________________________________________________________________ #
# __________________________________________________________________________________________________ #
# Define the VAE
d_latent = 4
vae = VAE(d_obs, d_latent=d_latent)
vae_trainer = TrainVAE(vae, step_size=8e-4, momentum_mass=0.6)


#__________________________________________________________________________________________________ #
# __________________________________________________________________________________________________ #
# Set up the training loop
enc_init_rng, dec_init_rng = random.split(random.PRNGKey(2))

_, init_encoder_params = vae.encoder_init(enc_init_rng, (-1, d_obs))
_, init_decoder_params = vae.decoder_init(dec_init_rng, (-1, d_latent))
init_params = init_encoder_params, init_decoder_params
opt_state = vae_trainer.opt_init(init_params)
itercount = itertools.count()

print("\nStarting VAE training...")

num_epochs = 10
batch_size = 128

num_train = X_train.shape[0]
num_complete_batches, leftover = divmod(num_train, batch_size)
num_batches = num_complete_batches + bool(leftover)

batches = data_stream(X_train, y_train, num_train, batch_size, num_batches)

test_rng = random.PRNGKey(0)
pbar = trange(num_epochs)
elbo_s = []
for epoch in pbar:
    start_time = time.time()
    epoch_rng = random.PRNGKey(epoch)
    for _ in range(num_batches):
        batch, _ = next(batches)
        opt_state = vae_trainer.update(epoch_rng, next(itercount), opt_state, batch)
        params = vae_trainer.get_params(opt_state)
    elbo_rng, data_rng = random.split(test_rng)
    binarized_test = random.bernoulli(data_rng, X_test)
    l = vae_trainer.elbo(elbo_rng, vae_trainer.get_params(opt_state), binarized_test)
    elbo_s.append(l)
    pbar.set_description("ELBO: {:.1f}".format(l))
    epoch_time = time.time() - start_time

    params = vae_trainer.get_params(opt_state)

plt.plot(elbo_s)
plt.xlabel('Epoch')
plt.ylabel('ELBO')
plt.title('VAE ELBO')
plt.show()

#__________________________________________________________________________________________________ #
# __________________________________________________________________________________________________ #
# Test trained VAE
vae_params = vae_trainer.get_params(opt_state)
# Plot a few reconstructions of images

# # Run PCA on the embeddings to get dimensions of maximal variance
# from sklearn.decomposition import PCA
# Z = vae.encoder(vae_params[0], X_train[:200])[0]
# pca = PCA(2).fit(Z)
#
# # Plot the embeddings for various digit classes
# plt.figure(figsize=(12, 12))
# for i in range(10):
#     inds = np.where(np.argmax(y_train, axis=-1) == i)[0][:200]
#     z = pca.transform(vae.encoder(vae_params[0], X_train[inds])[0])
#     plt.plot(z[:, 0], z[:, 1], 'o', alpha=0.1, label="{:d}".format(i))
# plt.legend()
# plt.show()

### ----------------------------------------------------------------------------------------------- ###
### ---------------------------------- Classifier Training ---------------------------------------- ###
### ----------------------------------------------------------------------------------------------- ###

# __________________________________________________________________________________________________ #
# __________________________________________________________________________________________________ #
# load the models
classifier = Classifier(d_latent=d_latent)

# __________________________________________________________________________________________________ #
# __________________________________________________________________________________________________ #
# Set up the optimiser
step_size = 0.001
num_epochs = 10
momentum_mass = 0.9

encode = lambda x: vae.encoder(vae_params[0], x)[0]
classifier_trainer = TrainClassifier(encode, classifier, step_size, momentum_mass)

# __________________________________________________________________________________________________ #
# __________________________________________________________________________________________________ #
# Set up the training loop
rng = random.PRNGKey(0)
_, init_classifier_params = classifier.classifier_init(rng, (-1, d_latent))
opt_state = classifier_trainer.opt_init(init_classifier_params)
itercount = itertools.count()

print("\nStarting classifier training...")
test_rng = random.PRNGKey(0)
pbar = trange(num_epochs)
losses = []
for epoch in pbar:
    start_time = time.time()
    start_time = time.time()
    sample_rng = random.PRNGKey(epoch)
    for _ in range(num_batches):
      opt_state = classifier_trainer.update(next(itercount), opt_state, next(batches))
    epoch_time = time.time() - start_time
    l = classifier_trainer.loss(classifier_trainer.get_params(opt_state), (X_test[:1000], y_test[:1000]))
    losses.append(l)
    pbar.set_description("Loss: {:.3f}".format(l))

plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Classifier Loss')
plt.show()

classifier_params = classifier_trainer.get_params(opt_state)
train_acc = classifier_trainer.accuracy(classifier_params, (X_train, y_train))
test_acc = classifier_trainer.accuracy(classifier_params, (X_test, y_test))
print(f"Epoch {epoch} in {epoch_time:0.2f} sec")
print(f"Training set accuracy {train_acc}")
print(f"Test set accuracy {test_acc}")

plot_examples(X_test,
              lambda x: np.argmax(classifier_trainer.classify(x, classifier_params)),
              lambda x: vae._reconstruct(x, vae_params)
              )


ckpt = {'model': classifier_trainer, 'classifier_params': classifier_params, 'vae_params': vae_params,
        'd_latent': d_latent, 'd_obs': d_obs}

def save(filename):
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    save_args = orbax_utils.save_args_from_target(ckpt)
    orbax_checkpointer.save(f'tmp/orbax/{filename}', ckpt, save_args=save_args)