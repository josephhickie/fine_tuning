"""
Created on 02/10/2023
@author jdh
"""

import numpy as np
import matplotlib
from jax import grad

from sklearn.neighbors import KernelDensity
from tqdm import tqdm

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from fine_tuning.models.capacitance.rust import get_location_of_first_triple_point
from sklearn.mixture import GaussianMixture
from fine_tuning.models.capacitance.jax_backend import constant_capacitance, constant_capacitance_old
from fine_tuning.algorithms.vae.utils import data_stream, plot_examples, load_data, load_stability_data, fetch_dataset
import jax.numpy as jnp
from jax import jit
from fine_tuning.models.capacitance.jax_backend import sensor
from scipy.signal import argrelextrema

(X_train, X_test, y_train_b, y_test_b) = load_stability_data()

#
# one = X_train[20, ...].reshape(62, 62)
#
# plt.figure()
# plt.imshow(one.T, origin='lower')
# plt.show()

x, y = fetch_dataset()
y[y == 2] = 1


def get(i):
    return x[i, ...]


def fit(data, n_components=4, plot=False):
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
    return residual, model


do = lambda i, n=4: fit(get(i), n)

from fine_tuning.models.capacitance import do2d

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

initial_params = np.array([
    cdd_diag_ratio, c_dg_0, c_dg_1, c_dg_2, c_dg_3,
    x_shift, y_shift, contrast_0, contrast_1, offset,
    gamma, x0
])

i2 = initial_params * (1 + 0.3 * np.random.rand(initial_params.size))
copy_i2 = jnp.copy(i2)


def e(params):
    return jnp.sum(((do2d_(initial_params) - do2d_(params)) ** 2))


grad_X = jit(e)


def fit_with_gmm(input_data, initial_params, method='Nelder-Mead'):
    residual, gmm_version_of_data = fit(input_data)

    plt.figure()
    plt.imshow(gmm_version_of_data.reshape(62, 62).T, origin='lower')
    plt.show()

    e = lambda params: error(gmm_version_of_data.flatten(), do2d_(params).flatten())
    result = minimize(e, initial_params, method=method)

    e2 = lambda params: error(input_data.T.flatten(), do2d_(params).flatten())

    result2 = minimize(e2, result.x, method=method)

    plt.figure()
    plt.imshow(input_data.reshape(62, 62).T, origin='lower')
    plt.show()

    plt.figure()
    plt.imshow(do2d_(result2.x).T, origin='lower')
    plt.show()

    plt.figure()
    plt.title('original fit')
    plt.imshow(input_data.reshape(62, 62).T, origin='lower')
    # plt.imshow(do2d_(result2.x).T, origin='lower', alpha=0.2, cmap='Greys')
    plt.imshow(do2d_(result.x).T, origin='lower', alpha=0.2)
    plt.show()

    return result, result2, e2


@jit
def do2d_(params):
    return do2d(*params)


def make_cdd_from_ratio(ratio):
    return np.array([
        [1, -1 / ratio],
        [-1 / ratio, 1]
    ])


def make_cdg_from_params(params):
    return np.array([[params[1], params[2]], [params[3], params[4]]])


# plt.figure()
# plt.imshow(do2d_(initial_params).T, origin='lower')

residual, model = fit(get(44), 4)

from scipy.optimize import minimize

e2 = lambda params: error(get(1100).T.flatten(), do2d_(params).flatten())

e2_grad = grad(e2)


def error(a, b):
    error = np.sum((a - b) ** 2)
    return error


def get_triple(params):
    cdd_inv = np.linalg.inv(make_cdd_from_ratio(params[0]))
    c_dg = make_cdg_from_params(params)

    x_shift = params[5]
    y_shift = params[6]

    shift = np.array([x_shift, y_shift])

    location = get_location_of_first_triple_point(cdd_inv, c_dg)

    return shift, location


x_amplitude = 1
y_amplitude = 1
x_res = 62
y_res = 62

v_g = jnp.stack(
    jnp.meshgrid(
        jnp.linspace(-x_amplitude / 2, x_amplitude / 2, x_res),
        jnp.linspace(-y_amplitude / 2, y_amplitude / 2, y_res)
    ), axis=-1) + jnp.array([x_shift, y_shift])

c_dg = - jnp.array([
    [c_dg_0, c_dg_1],
    [c_dg_2, c_dg_3]
])

cdd_diag = 1.
cdd_off_diag = - 1 / cdd_diag_ratio

cdd = jnp.array([[cdd_diag_ratio, -1], [-1, cdd_diag_ratio]])
cdd_inv = jnp.linalg.inv(cdd)

state_contrast = jnp.array([contrast_0, contrast_1])


def kde_data(data, scale=25, score_sample_rate=1, plot=False, **kwargs):
    """

    :param data:
    :param scale:
    :param score_sample_rate:
    :param plot:
    :param kwargs:
    :return:
    """
    data = data * scale
    kde = KernelDensity(kernel='gaussian', **kwargs).fit(data.reshape(-1, 1))

    s = np.linspace(0, scale, scale * score_sample_rate)
    e = kde.score_samples(s.reshape(-1, 1))
    mi, ma = argrelextrema(e, np.less)[0], argrelextrema(e, np.greater)[0]

    if plot:
        plt.figure()
        plt.imshow(data.reshape(62, 62).T, origin='lower')
        plt.show()

        plt.figure()
        plt.plot(
            # s[:mi[0] + 1], e[:mi[0] + 1], 'r',
            #      s[mi[0]:mi[1] + 1], e[mi[0]:mi[1] + 1], 'g',
            #      s[mi[1]:mi[2] + 1], e[mi[1]:mi[2] + 1], 'b',
            #      s[mi[2]:], e[mi[2]:], 'k',
            s[ma], e[ma], 'go',
            s[mi], e[mi], 'ro')
        plt.plot(s, e)

    if len(mi) == 3 and len(ma) == 4:
        # print('four states')
        return 3
    elif len(mi) == 2 and len(ma) == 3:
        # print('three states')
        return 3
    elif len(mi) == 1 and len(ma) == 2:
        # print('two states')
        return 1
    else:
        return 0
        # print('one state / noise')

    # plt.figure()
    # plt.plot(s, e)


def check(i, plot=False, scale=30, sample_rate=3):
    guess = kde_data(x[i], scale=scale, score_sample_rate=sample_rate, plot=plot)
    # print(y[i], guess)
    if (guess == y[i]):
        return 1
    else:
        return 0


def check_all(ids, scale, sample_rate):
    correct = []
    for id in tqdm(ids):
        correct.append(check(id, scale=scale, sample_rate=sample_rate, plot=False))
    return np.array(correct)
