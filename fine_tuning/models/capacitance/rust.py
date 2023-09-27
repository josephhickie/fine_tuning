"""
Created on 25/09/2023
@author jdh
"""

import rusty_capacitance_model_core as r
import jax.numpy as np

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from itertools import combinations


def make_cdd(diag, off_diag):

    # off diag must be negative, diag positive

    return np.array([
        [diag, off_diag],
        [off_diag, diag]
    ])

def make_cdd_inv(diag, off_diag):
    cdd = make_cdd(diag, off_diag)
    return np.linalg.inv(cdd)

def triple_point(n, cdd_inv, c_dg, case=1):

    n00 = np.array([[n], [n]])
    n01 = np.array([[n], [n - case]])
    n10 = np.array([[n - case], [n]])

    combos = list(combinations([n00, n01, n10], r=2))

    M = []
    b = []

    for n, m in combos:
        u = 2 * (n.T - m.T) @ cdd_inv @ c_dg.T
        M.append(u.squeeze())

        p = n.T @ cdd_inv @ n - m.T @ cdd_inv @ m
        b.append(p.squeeze())

    M = np.stack(M)
    b = np.array(b)

    m_inv = np.linalg.pinv(M)

    return (m_inv @ b)[::-1]

def get_location_of_first_triple_point(cdd_inv, c_dg):


    p1 = triple_point(1, cdd_inv, c_dg, case=-1)
    p2 = triple_point(2, cdd_inv, c_dg)

    return p1, p2

def stability_diagram_around_triple(cdd_inv, c_dg, x_amplitude, y_amplitude, threshold=1):

    p1, p2 = get_location_of_first_triple_point(cdd_inv, c_dg)
    centre = (p1 + p2) / 2

    v_g = np.stack(
        np.meshgrid(
            np.linspace(-x_amplitude / 2, x_amplitude / 2, n),
            np.linspace(-y_amplitude / 2, y_amplitude / 2, m)
        ), axis=-1
    )

    v_g = v_g + centre
    transition = np.array([p1, p2])
    transition = transition - np.mean(transition)

    out = r.ground_state_open(
        np.reshape(v_g, (-1, 2)), c_dg, cdd_inv, threshold
    )

    return out, transition


#
# cdd = make_cdd(1, -0.2)
# cdd_inv = np.linalg.inv(cdd)
#
# n = 62
# m = 62
#
# c_dg = - np.array([
#     [0.6, 0.1],
#     [0.1, 0.6]
# ])
#
# x_amplitude = 2
# y_amplitude = 2
#
# out, transition = stability_diagram_around_triple(cdd_inv, c_dg, x_amplitude, y_amplitude)
#
#
# plt.imshow(np.sum(out.reshape(n, m, -1), axis=-1).T, origin='lower', extent=(-x_amplitude/2, x_amplitude/2, -y_amplitude/2, y_amplitude/2))
# plt.plot(*transition.T, color='black')
# plt.show()
#




