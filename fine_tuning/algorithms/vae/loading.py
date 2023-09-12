"""
Created on 11/09/2023
@author jdh
"""


import jax.numpy as np
from jax import jit
import matplotlib.pyplot as plt


def normalise(data):

    return (data - np.min(data)) / (data.max() - data.min())

from .vae import VAE
from .classifier import Classifier
from .utils import plot_examples, load_stability_data

import orbax.checkpoint
import pickle
# X_train, X_test, y_train, y_test = load_stability_data()


def orbax_to_numpy_save(filename):

    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    raw_restored = orbax_checkpointer.restore(filename)

    raw_restored = dict(raw_restored)

    with open(filename + '/np_checkpoint.pickle', 'wb') as handle:
        pickle.dump(raw_restored, handle, protocol=pickle.HIGHEST_PROTOCOL)


    # d_latent = raw_restored.get('d_latent')
    # d_obs = raw_restored.get('d_obs')
    # vae_params = raw_restored.get('vae_params')
    # classifier_params = raw_restored.get('classifier_params')



def model_file_to_classifier_fn(filename):

    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    raw_restored = orbax_checkpointer.restore(filename)
    d_latent = raw_restored.get('d_latent')
    d_obs = raw_restored.get('d_obs')
    vae_params = raw_restored.get('vae_params')
    classifier_params = raw_restored.get('classifier_params')

    vae = VAE(d_obs, d_latent=d_latent)
    classifier = Classifier(d_latent=d_latent)

    encode_fn = lambda x: vae.encoder(vae_params[0], x)[0]

    @jit
    def classify(X):
        z = encode_fn(X)
        return np.argmax(classifier.predict(classifier_params, z))

    return classify

def np_model_file_to_classifier_fn(filename):

    with open(filename, 'rb') as handle:
        raw_restored = pickle.load(handle)

    d_latent = raw_restored.get('d_latent')
    d_obs = raw_restored.get('d_obs')
    vae_params = raw_restored.get('vae_params')
    classifier_params = raw_restored.get('classifier_params')

    vae = VAE(d_obs, d_latent=d_latent)
    classifier = Classifier(d_latent=d_latent)

    encode_fn = lambda x: vae.encoder(vae_params[0], x)[0]

    @jit
    def classify(X):
        z = encode_fn(X)
        return np.argmax(classifier.predict(classifier_params, z))

    return classify

#
# example = lambda *args: plot_examples(X_test,
#               lambda x: np.argmax(classifier_trainer.classify(x, classifier_params)),
#               lambda x: vae._reconstruct(x, vae_params)
# )
#
# example()
#
# import sys
#
# sys.path.append('/home/jdh/PycharmProjects/qgor_qm')
# from qgor_simulation import Station
#
# station = Station()
#
#
# def plot(i, name='PCA_0'):
#
#     data = normalise(station.database.load(i).mm_r.get(name))
#
#     plt.figure()
#     plt.imshow(data)
#     plt.show()
#     print(np.argmax(classifier_trainer.classify(data.flatten(), classifier_params)),)
