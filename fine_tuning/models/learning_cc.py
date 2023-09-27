"""
Created on 26/09/2023
@author jdh
"""

import numpy as np
import jax.numpy as jnp
import matplotlib
import dm_pix as pix
from jax import grad, value_and_grad
import jax
from tqdm import tqdm

from scipy.ndimage import gaussian_filter
matplotlib.use('TkAgg')
from scipy.optimize import minimize
import matplotlib.pyplot as plt

from capacitance import do2d
from capacitance.jax_backend import constant_capacitance


x_res = 62
y_res = 62
x_amplitude = 1.
y_amplitude = 1.

v_g = jnp.stack(
    jnp.meshgrid(
        jnp.linspace(-x_amplitude / 2, x_amplitude / 2, x_res),
        jnp.linspace(-y_amplitude / 2, y_amplitude / 2, y_res)
    ), axis=-1)

c_dg = - jnp.array([
    [1, 0.1],
    [0.1, 1]
])

cdd = jnp.array([[1, -0.1], [-0.1, 1]])
cdd_inv = jnp.linalg.inv(cdd)




def normalise(data):
    # return data
    return (data - data.min()) / (data.max() - data.min())

params_max = jnp.array([
    10, 1.5, 0.9, 0.9, 1.5, 0.5, 0.5, 2, 2, 2, 2, 2
])

params_min = jnp.array([
    1.1, 0.1, 0.1, 0.1, 0.1, -0.5, -0.5, 0.1, 0.1, -2, 0, -2
])

def random_params():

    scales = np.random.rand(params_max.size)
    vals =  params_min + scales * (params_max - params_min)
    return vals


def random_data():
    return np.abs(normalise(do2d(*random_params()) + 0.01 * np.random.rand(62, 62) ) - 1)

def plot_random_data():
    dat = random_data()
    plt.figure()
    plt.imshow(dat.T, origin='lower')
    plt.show()
    return dat


cdd_diag_ratio = 8
c_dg_0 = 1
c_dg_1 = 0.05
c_dg_2 = 0.58
c_dg_3 = 1
x_shift = 0.2
y_shift = -0.4
contrast_0 = 1.2
contrast_1 = 1.3
offset = 1
gamma = 1
x0 = 1

initial_params = jnp.array([
    cdd_diag_ratio, c_dg_0, c_dg_1, c_dg_2, c_dg_3,
    x_shift, y_shift, contrast_0, contrast_1, offset,
    gamma, x0
])

bad_params = jnp.array([
    2, 1, 0.1, 0.3, 1, -0.3, 0.5, 0.7, 0.7, 0., 1, 0
])

data_file = '/home/jdh/Documents/vae_training/triple_with_compensation/16940848599616.npy'
data_ = np.load(data_file)
data = gaussian_filter(data_, sigma=1)

simulated_data = do2d(*initial_params)
bad_data = do2d(*bad_params)


plt.figure()
plt.imshow(data.T, origin='lower')
plt.show()

plt.figure()
plt.imshow(simulated_data.T, origin='lower')
plt.show()


# plt.figure()
# plt.imshow(bad_data.T, origin='lower')
# plt.show()


one = jnp.ones_like(initial_params)

def generate(params):
    return do2d(*params)


def get_loss_for_params(_params):
    simulated_data_ = (generate(_params))
    return loss(data, simulated_data_)


def loss(data, simulated_data):
    return jnp.sum((data - simulated_data) ** 2)

def lorentz(x, gamma, x0=0):
    return (gamma / 2) ** 2 / ((x - x0) ** 2 + (gamma / 2) ** 2)


def plot_for_params(params):
    data = generate(params)
    plt.figure()
    plt.imshow(data.T, origin='lower')
    plt.show()

@jax.jit
def get_grads(params):
    return grad(l_for_p)(params)



def ssim(a, b, **kwargs):
    return pix.ssim(jnp.expand_dims(a, axis=0), jnp.expand_dims(b, axis=0), **kwargs)


def mse(a, b):

    return np.sum(10 * ((a - b))**2) / a.size


def l_for_p(_params):
    # params = initial_params * _params

    loss = mse(generate(initial_params), do2d(*_params))

    return loss