"""
Created on 10/10/2023
@author jdh
"""



import jax.numpy as jnp
from fine_tuning.models.capacitance import do2d
from jax import jit, vmap
import numpy as np

from fine_tuning.models.separate import generate_data


@jit
def do2d_(params):
    return do2d(*params)


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

initial_params = jnp.array([
    cdd_diag_ratio, c_dg_0, c_dg_1, c_dg_2, c_dg_3,
    x_shift, y_shift, contrast_0, contrast_1, offset,
    gamma, x0
])

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



n_training_samples = 20000
n_test_samples = 10000
params_max = jnp.array([
    10, 2, 0.4, 0.4, 2, 1, 1, 2, 2, 3, 3, 3
])

params_min = jnp.array([
    2, 0.5, 0, 0, 0.5, -1, -1, 0.3, 0.3, 1, 1, 1
])


(X_train, y_train), (X_test, y_test) = generate_data(n_training_samples, n_test_samples, params_min, params_max)


