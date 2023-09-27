"""
Created on 25/09/2023
@author jdh
"""


import jax.numpy as jnp

from .jax_backend import ground, make_cdd, sensor
from .rust import get_location_of_first_triple_point


def do2d(
        cdd_diag_ratio, c_dg_0, c_dg_1, c_dg_2, c_dg_3,
        x_shift, y_shift, contrast_0, contrast_1, offset,
        gamma, x0, x_res=62, y_res=62
):

    # assert c_dg_0 > 0.
    # assert c_dg_1 > 0.
    # assert c_dg_2 > 0.
    # assert c_dg_3 > 0.
    #
    # assert jnp.abs(x_shift) <= 1.
    # assert jnp.abs(y_shift) <= 1.


    # the x_amp and y_amp parameters are only important in their ratio to the c_dg matrix,
    # so i fix them here. The learnt c_dg_x parameters will then just be learnt in scale.
    x_amplitude = 1.
    y_amplitude = 1.

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

    cdd = make_cdd(cdd_diag, cdd_off_diag)
    cdd_inv = jnp.linalg.inv(cdd)

    p1, p2 = get_location_of_first_triple_point(cdd_inv, c_dg)
    centre = (p1 + p2) / 2.

    # shift the window to centre it on the triple point, then have another parameter to move the centre from
    # the triple point itself to anywhere within the window as it would be if the triple point was centred
    v_g = v_g + centre + (jnp.array([x_shift, y_shift]) * jnp.array([x_amplitude, y_amplitude]) / 2.)

    state_contrast = jnp.array([contrast_0, contrast_1])
    return sensor(v_g, cdd_inv, c_dg, state_contrast, offset, gamma, x0)
    # return ground(v_g, cdd_inv, c_dg, state_contrast)

#
# _cdd_ratio = 10
# _x_amp = 1
# _y_amp = 1
# _c_dg = -jnp.array([[1, 0], [0, 1]])
# _x_shift = 0
# _y_shift = 0
#
# v_g = jnp.stack(
#         jnp.meshgrid(
#             jnp.linspace(-_x_amp / 2, _x_amp / 2, resolution[0]),
#             jnp.linspace(-_y_amp / 2, _y_amp / 2, resolution[1])
#         ), axis=-1
#     )
#
#
# plt.figure()
# ground_state = do2d(_cdd_ratio, _x_amp, _y_amp, _c_dg, _x_shift, _y_shift, v_g)
# plt.imshow(ground_state.T, origin='lower')
#
#
#
# cdd_ratio = lambda x: plt.imshow(do2d(x, _x_amp, _y_amp, _c_dg, _x_shift, _y_shift).T, origin='lower')
# x_amp = lambda x: plt.imshow(do2d(_cdd_ratio, x, _y_amp, _c_dg, _x_shift, _y_shift).T, origin='lower')
# y_amp = lambda x: plt.imshow(do2d(_cdd_ratio, _x_amp, x, _c_dg, _x_shift, _y_shift).T, origin='lower')
# c_dg = lambda x: plt.imshow(do2d(_cdd_ratio, _x_amp, _y_amp, x, _x_shift, _y_shift).T, origin='lower')
# x_shift = lambda x: plt.imshow(do2d(_cdd_ratio, _x_amp, _y_amp, _c_dg, x, _y_shift).T, origin='lower')
# y_shift = lambda x: plt.imshow(do2d(_cdd_ratio, _x_amp, _y_amp, _c_dg, _x_shift, x).T, origin='lower')
#


# p1, p2 = get_location_of_first_triple_point(cdd_inv, c_dg)
# centre = jnp.mean([p1, p2])
#
# a = ground(v_g, )
#
#
#
#
#
# cdd = jnp.array([
#     [1, -0.1],
#     [-0.1, 1]
# ])
#
# cdd_inv = jnp.linalg.inv(cdd)
#
# x_amplitude = 10
# y_amplitude = 10
#
# n = 100
# m = 100
#
# v_g = jnp.stack(
#     jnp.meshgrid(
#         jnp.linspace(-x_amplitude / 2, x_amplitude / 2, n),
#         jnp.linspace(-y_amplitude / 2, y_amplitude / 2, m)
#     ), axis=-1)
# )
#
# c_dg = - jnp.array([
#     [1, 0.1],
#     [0.1, 1]
# ])
#
# @jax.jit
# def constant_capacitance(v_g, cdd_inv, c_dg, max_charges=100):
#
#     n_continuous = jnp.einsum('ij, abj', c_dg, v_g)
#     floor_ceiling_list = product([jnp.floor, jnp.ceil], repeat=n_continuous.shape[2])
#
#     n_discrete = jnp.stack([jnp.stack([operator(data)
#                                      for operator, data
#                                      in zip(operations, jnp.rollaxis(n_continuous, 2))], axis=2)
#                            for operations
#                            in floor_ceiling_list], axis=0)
#
#     n_discrete = jnp.clip(n_discrete, a_min=0, a_max=max_charges)
#
#     v = n_discrete - n_continuous
#
#     u = jnp.einsum('abcd, de, abce -> abc', v, cdd_inv, v)
#     # u = v.T.dot(cdd_inv)*v.sum(axis=-1)
#
#     return n_discrete, n_continuous, u
#
# @jax.jit
# def ground_state(N_discrete, U, state_contrast):
#
#     arg_min = jnp.expand_dims(jnp.argmin(U, axis=0), axis=(0, 3))
#     ground_state = jnp.take_along_axis(N_discrete, arg_min, 0)
#
#     return (ground_state * state_contrast[jnp.newaxis, jnp.newaxis, jnp.newaxis, :]).sum(axis=3).squeeze()
#
# state_contrast = jnp.array([1, 1])
#
# @jax.jit
# def ground(v_g, cdd_inv, c_dg, state_contrast):
#     n_discrete, n_continuous, u = constant_capacitance(v_g, cdd_inv, c_dg)
#     a = ground_state(n_discrete, u, state_contrast)
#     return a
#
