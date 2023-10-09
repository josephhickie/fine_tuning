"""
Created on 26/09/2023
@author jdh
"""

import jax.numpy as jnp
from itertools import product
import matplotlib.pyplot as plt
import jax
import time

import numpy as np


def lorentz(x, gamma, x0=0):
    return (gamma / 2) ** 2 / ((x - x0) ** 2 + (gamma / 2) ** 2)

# @jax.jit
def constant_capacitance_old(v_g, cdd_inv, c_dg, max_charges=3):
    n_continuous = jnp.einsum('ij, abj', c_dg, v_g)


    floor_ceiling_list = product([jnp.floor, jnp.ceil], repeat=n_continuous.shape[2])
    # floor_ceiling_list = product([lambda x: x, lambda x: x], repeat=n_continuous.shape[2])

    n_discrete = jnp.stack([jnp.stack([operator(data)
                                     for operator, data
                                     in zip(operations, jnp.rollaxis(n_continuous, 2))], axis=2)
                           for operations
                           in floor_ceiling_list], axis=0)

    n_discrete = jnp.clip(n_discrete, a_min=0, a_max=max_charges)
    v = n_discrete - n_continuous
    u = jnp.einsum('abcd, de, abce -> abc', v, cdd_inv, v)

    print(u.shape)
    print(n_discrete.shape)
    return n_discrete, n_continuous, u


def floor(x, alpha=10):
    # return (1. / 1. + jnp.exp(-alpha * (x - jnp.floor(x) )))
    return x - jax.nn.relu(x - 0.5)


# @jax.jit
def constant_capacitance(v_g, cdd_inv, c_dg, max_charges=100):
    n_continuous = jnp.einsum('ij, abj', c_dg, v_g)


    n_discrete = floor(n_continuous)

    n = cdd_inv.shape[0]

    floor_ceil_list = jnp.stack(
        jnp.meshgrid(
        *(n * [jnp.arange(n)])
    ), axis=-2).reshape(-1, n)

    floor_ceil_list = jnp.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])

    n_discrete = n_discrete[jnp.newaxis, ...] + floor_ceil_list[:, jnp.newaxis, jnp.newaxis, :]

    n_discrete = jnp.clip(n_discrete, a_min=0, a_max=max_charges)
    v = n_discrete - n_continuous
    u = jnp.einsum('abcd, de, abce -> abc', v, cdd_inv, v)

    # print(u.shape)
    # print(n_discrete.shape)
    # print(n_continuous.shape)
    # return jnp.sum(n_discrete)
    return n_discrete, n_continuous, u




# @jax.jit
def ground_state(N_discrete, U, state_contrast):
    arg_min = jnp.expand_dims(jnp.argmin(U, axis=0), axis=(0, 3))
    ground_state = jnp.take_along_axis(N_discrete, arg_min, 0)
    return (ground_state * state_contrast[jnp.newaxis, jnp.newaxis, jnp.newaxis, :]).sum(axis=3).squeeze()

# @jax.jit
def ground(v_g, cdd_inv, c_dg, state_contrast):
    n_discrete, n_continuous, u = constant_capacitance(v_g, cdd_inv, c_dg)
    a = ground_state(n_discrete, u, state_contrast)
    return a



def make_cdd(diag, off_diag):

    # off diag must be negative, diag positive

    return jnp.array([
        [diag, off_diag],
        [off_diag, diag]
    ])


# @jax.jit
def sensor_state(N_discrete, U, state_contrast, sensor_offset):
    # arg_min = jnp.argmin(U, axis=0, keepdims=True)[..., jnp.newaxis]
    arg_min = jnp.expand_dims(jnp.argmin(U, axis=0), axis=(0, 3))
    ground_state = jnp.take_along_axis(N_discrete, arg_min, 0)

    return sensor_offset + (
            ground_state * jnp.expand_dims(state_contrast, axis=(0, 1, 2))
    ).sum(axis=-1).squeeze()


# @jax.jit
def sensor_energy(v_g, cdd_inv, c_dg, state_contrast, sensor_offset):
    n_discrete, n_continuous, u = constant_capacitance(v_g, cdd_inv, c_dg)
    a = sensor_state(n_discrete, u, state_contrast, sensor_offset)
    return a


def sensor(v_g, cdd_inv, c_dg, state_contrast, sensor_offset, lorentz_gamma, lorentz_x0):
    a = sensor_energy(v_g, cdd_inv, c_dg, state_contrast, sensor_offset)
    return lorentz(a, lorentz_gamma, lorentz_x0)
