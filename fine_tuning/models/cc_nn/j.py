"""
Created on 10/10/2023
@author jdh
"""
import jax
import optax
from tqdm import tqdm

"""
Double dot example
"""

import matplotlib
matplotlib.use('TkAgg')
import jax.numpy as jnp
import matplotlib.pyplot as plt
import jax.nn as nn
import numpy as np
from jax import grad, jit
# from skimage.metrics import structural_similarity as ssim
from dm_pix import ssim

from fine_tuning.models import gmm_fit

def energies(cdd_inv, cgd, Vg):
    n_list = jnp.array([
        [0, 0],
        [1, 0],
        [0, 1],
        [1, 1],
        [2, 0],
        [0, 2]
    ])

    v = jnp.einsum('ij, ...j -> ...i', cgd, Vg)
    delta = n_list[jnp.newaxis, jnp.newaxis, :] - v[..., jnp.newaxis, :]
    # computing the free energy of the change configurations
    F = jnp.einsum('...i, ij, ...j', delta, cdd_inv, delta) / 2
    return F

def do2d(params):
    return do2d_grads(*params)


def lorentz(x, gamma, x0=0):
    return (gamma / 2) ** 2 / ((x - x0) ** 2 + (gamma / 2) ** 2)

def do2d_grads(
        cdd_diag_ratio, c_dg_0, c_dg_1, c_dg_2, c_dg_3,
        x_shift, y_shift, contrast_0, contrast_1, offset,
        gamma, x0, x_res=62, y_res=62
):

    x_amplitude = 2.
    y_amplitude = 2.

    v_g = jnp.stack(
        jnp.meshgrid(
            jnp.linspace(-x_amplitude / 2, x_amplitude / 2, x_res),
            jnp.linspace(-y_amplitude / 2, y_amplitude / 2, y_res)
        ), axis=-1)

    c_dg = - jnp.array([
        [c_dg_0, c_dg_1],
        [c_dg_2, c_dg_3]
    ])

    cdd_diag = 1.
    cdd_off_diag = - 1 / cdd_diag_ratio

    cdd = jnp.array([
        [cdd_diag, cdd_off_diag],
        [cdd_off_diag, cdd_diag]
    ])

    cdd_inv = jnp.linalg.inv(cdd)

    # p1, p2 = get_location_of_first_triple_point(cdd_inv, c_dg)
    # centre = (p1 + p2) / 2.

    # print(centre)
    centre = jnp.array([0., 0.])


    # shift the window to centre it on the triple point, then have another parameter to move the centre from
    # the triple point itself to anywhere within the window as it would be if the triple point was centred
    v_g = v_g + centre + (jnp.array([x_shift, y_shift]) * jnp.array([x_amplitude, y_amplitude]) / 2.)

    n_list = jnp.array([
        [0, 0],
        [1, 0],
        [0, 1],
        [1, 1],
        [2, 0],
        [0, 2]
    ])

    v = jnp.einsum('ij, ...j -> ...i', c_dg, v_g)
    delta = n_list[jnp.newaxis, jnp.newaxis, :] - v[..., jnp.newaxis, :]
    # computing the free energy of the change configurations
    F = jnp.einsum('...i, ij, ...j', delta, cdd_inv, delta) / 2

    T = 0.001
    p = nn.softmax(-F / T)

    colours = jnp.array([0, 1, 2, 3, 4, 5])

    n = jnp.einsum('...i, ...i', p, colours)
    state_contrast = jnp.array([contrast_0, contrast_1])


    return lorentz(n, gamma, x0)

    # return sensor(v_g, cdd_inv, c_dg, state_contrast, offset, gamma, x0)
    # return ground(v_g, cdd_inv, c_dg, state_contrast)


cdd_diag_ratio = 5
c_dg_0 = 1
c_dg_1 = 0.1
c_dg_2 = 0.1
c_dg_3 = 1
x_shift = 0.
y_shift = 0.
contrast_0 = -0.8
contrast_1 = -1.3
offset = 0
gamma = 2
x0 = -9

initial_params = jnp.array([
    cdd_diag_ratio, c_dg_0, c_dg_1, c_dg_2, c_dg_3,
    x_shift, y_shift, contrast_0, contrast_1, offset,
    gamma, x0
])


init = jnp.array([
    cdd_diag_ratio, c_dg_0, c_dg_1, c_dg_2, c_dg_3,
    x_shift, y_shift, contrast_0, contrast_1, offset,
    gamma, x0
]) * (1 + np.random.uniform(0.2, size=initial_params.shape))

real_data = jnp.load('/home/jdh/Documents/vae_training/triple_with_compensation/169410163977659.npy')

residual, model, gm = gmm_fit(
    real_data,
    plot=False
)


def learning(data, params, opt_fn, opt_state, steps=10000):

    @jit
    def f(theta):
        return - ssim(data[jnp.newaxis, ...], do2d(theta)[jnp.newaxis, ...], max_val=data.max())
        return jnp.sum((data - do2d(theta))**2)

    losses = []

    for i in range(steps):
        loss, grads = jax.value_and_grad(f)(params)
        updates, opt_state = opt_fn(grads, opt_state)
        params += updates
        losses.append(loss)
        if i % 100 == 0:
            print(f'{i}: {loss}')

    return jnp.stack(losses), params, opt_state

# grad_e = grad(e)


lr = 3e-2
adam = optax.adagrad(learning_rate=lr)

#
# for i in tqdm(range(1000)):
#     init -= lr * grad_e(init)
#
#     if i % 10 == 0:
#         print(e(init))

losses, params, opt_state = learning(model, initial_params, opt_fn=adam.update, opt_state=adam.init(init))


plt.figure()
plt.title('after grad desc')
plt.imshow(do2d(params).T, origin='lower')
plt.show()

plt.figure()
plt.title('real data')
plt.imshow(model.T, origin='lower')
plt.show()

plt.figure()
plt.title('original')
plt.imshow(do2d(initial_params).T, origin='lower')
plt.show()

plt.figure()
plt.plot(losses)
plt.show()