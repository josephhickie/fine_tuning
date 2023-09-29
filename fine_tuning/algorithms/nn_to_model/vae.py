import jax.numpy as np
from jax import random, jit, grad
from jax.example_libraries import optimizers

from jax.example_libraries import stax
from jax.example_libraries.stax import Dense, Relu, LogSoftmax, Sigmoid, FanOut, Softplus
from jax.scipy.stats import norm
from functools import partial

# from flax import linen as nn          # The Linen API




class VAE:
    """
    A simple VAE
    """
    def __init__(self, d_obs, n_dense_layers=2, d_latent=10, d_hidden=512):
        """

        :param d_obs:
        :param d_latent:
        :param d_hidden:
        """
        self.d_obs = d_obs
        self.d_latent = d_latent
        self.d_hidden = d_hidden
        self.n_dense_layers = n_dense_layers
        self._init_encoder_()
        self._init_decoder_()


    def _init_encoder_(self):
        dense_layers = [Dense(self.d_hidden), Relu]*self.n_dense_layers
        self.encoder_init, self.encoder = stax.serial(
                                            *dense_layers,
                                            FanOut(2),
                                            stax.parallel(Dense(self.d_latent),
                                                          stax.serial(Dense(self.d_latent), Softplus)))

    def _init_decoder_(self):
        dense_layers = [Dense(self.d_hidden), Relu]*self.n_dense_layers
        self.decoder_init, self.decoder = stax.serial(
                                            *dense_layers,
                                            Dense(self.d_obs), Sigmoid)


    def _reconstruct(self, X, params):
        encoder_params, decoder_params = params
        return self.decoder(decoder_params, self.encoder(encoder_params, X)[0])


# class ConVAE(nn.Module):
#
#     def __init__(self, d_obs, d_latent, d_hidden):
#         super().__init__(d_obs, d_latent, d_hidden)
#
#
#     def _init_encoder_(self):
#         pass
#
#     @nn.compact
#     def encoder(self, x):
#
#         x = nn.Conv(features=32, kernel_size=(3, 3))(x)
#         x = nn.relu(x)
#         x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
#         x = nn.Conv(features=64, kernel_size=(3, 3))(x)
#         x = nn.relu(x)
#         x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
#         x = x.reshape((x.shape[0], -1))  # Flatten
#         x = nn.Dense(features=256)(x)
#         x = nn.relu(x)
#         x = nn.Dense(features=10)(x)  # There are 10 classes in MNIST
#         return x
#
#     def _init_decoder_(self):
#         pass



class TrainVAE:
    """
    A class to train the VAE.
    """
    def __init__(self, vae, step_size):
        """

        :param vae:
        :param step_size:
        :param momentum_mass:
        """
        self.vae = vae
        self.step_size = step_size
        self._init_optimiser_()

    def _init_optimiser_(self):
        step_size = optimizers.exponential_decay(step_size=self.step_size, decay_steps=1000, decay_rate=0.9)
        self.opt_init, self.opt_update, self.get_params = optimizers.adam(step_size)
        #self.opt_init, self.opt_update, self.get_params = optimizers.momentum(self.step_size, mass=self.momentum_mass)

    # Define the loss function
    @partial(jit, static_argnums=(0,))
    def elbo(self, elbo_rng, params, batch):
        enc_params, dec_params = params


        # get mu, sigmasq output from encoder net using batch as input
        mu, sigmasq = self.vae.encoder(enc_params, batch)

        # sample noise from normal distribution
        epsilon = random.normal(elbo_rng, shape=mu.shape)  # (batch.shape[0], d_latent,))

        # calculate z from mu, sigmasq, and epsilon
        z = mu + np.sqrt(sigmasq) * epsilon

        # calculate log_q epsilon, and sigmasq.
        # hint: use norm.logpdf(epsilon) to calculate the log pdf of gaussian (from jax.scipy.stats)
        log_q = np.sum(norm.logpdf(z, mu, np.sqrt(sigmasq)), axis=1)

        # get p from decoder using z as input
        p = self.vae.decoder(dec_params, z)
        # calculate (log) bernoulli likelihood
        # and the (log) standard normal prior
        log_p = np.sum(np.log(p) * batch + np.log(1 - p) * (1 - batch), axis=1) \
                + np.sum(norm.logpdf(z), axis=1)
        elbo = np.mean(log_p - log_q)
        return elbo

    @partial(jit, static_argnums=(0,))
    def update(self, rng, i, opt_state, batch):
        elbo_rng, _ = random.split(random.fold_in(rng, i))
        data_rng, _ = random.split(random.fold_in(rng, i), 2)
        params = self.get_params(opt_state)
        binarized_batch = random.bernoulli(data_rng, batch)

        loss = lambda params: -self.elbo(elbo_rng, params, binarized_batch)
        g = grad(loss)(params)

        return self.opt_update(i, g, opt_state)
