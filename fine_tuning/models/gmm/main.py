"""
Created on 02/10/2023
@author jdh
"""

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from sklearn.mixture import GaussianMixture

from fine_tuning.algorithms.vae.utils import data_stream, plot_examples, load_data, load_stability_data

(X_train, X_test, y_train_b, y_test_b) = load_stability_data()


one = X_train[20, ...].reshape(62, 62)

plt.figure()
plt.imshow(one.T, origin='lower')
plt.show()

def get(i):
    return X_train[i, ...]

def fit(data, n_components=4):

    gm = GaussianMixture(n_components).fit(data.reshape(-1, 1))

    plt.figure()
    plt.imshow(data.reshape(62, 62).T, origin='lower')
    plt.show()


    model = gm.means_[gm.predict(data.reshape(-1, 1))].reshape(62, 62)


    plt.figure()
    plt.imshow(model.T, origin='lower')
    plt.show()

    residual = np.sum((data.flatten() - model.flatten())**2)
    return residual


do = lambda i, n=4: fit(get(i), n)