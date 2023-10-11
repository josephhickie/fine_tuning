"""
Created on 10/10/2023
@author jdh
"""

from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import numpy as np

def gmm_fit(data, n_components=4, plot=False):
    gm = GaussianMixture(n_components).fit(data.reshape(-1, 1))
    model = gm.means_[gm.predict(data.reshape(-1, 1))].reshape(62, 62)

    if plot:
        plt.figure()
        plt.imshow(data.reshape(62, 62).T, origin='lower')
        plt.show()

        plt.figure()
        plt.imshow(model.T, origin='lower')
        plt.show()

    residual = np.sum((data.flatten() - model.flatten()) ** 2)
    return residual, model, gm
