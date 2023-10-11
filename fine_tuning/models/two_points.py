"""
Created on 11/10/2023
@author jdh
"""


# import jax.numpy as np
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from skimage import transform as tf

from jax import grad, jit


def get_grad_and_offset(point1, point2):
    delta = (point1 - point2)
    m = delta[1] / delta[0]
    c = point1[1] - m * point1[0]

    return m, c

def full(x1, y1, x2, y2, bottom_lim, top_lim, left_lim, right_lim):

    point1 = np.array(
        [x1, y1]
    )

    point2 = np.array(
        [x2, y2]
    )


    x_amp = 1
    y_amp = 1
    res = 100

    x = np.linspace(-x_amp / 2, x_amp / 2, res)
    y = np.linspace(-y_amp / 2, y_amp / 2, res)

    X, Y = np.meshgrid(x, y)

    plane = np.stack(
        [X, Y], axis=0
    )

    plane2 = np.stack(
        [X, Y], axis=0
    )

    m, c = get_grad_and_offset(point1, point2)

    bl_m_v, bl_c_v = get_grad_and_offset(point1, np.array([bottom_lim, y[0]]))
    bl_m_h, bl_c_h = get_grad_and_offset(point1, np.array([x[0], left_lim]))

    tr_m_v, tr_c_v = get_grad_and_offset(point2, np.array([top_lim, y[-1]]))
    tr_m_h, tr_c_h = get_grad_and_offset(point2, np.array([x[-1], right_lim]))

    bl = 1 * np.all([X < (Y - bl_c_h) / bl_m_h, Y < X * bl_m_v + bl_c_v], axis=0)
    tr = 2 * np.all([X > (Y - tr_c_h) / tr_m_h, Y > X * tr_m_v + tr_c_v], axis=0)

    tl = 3. * np.all(np.array([(Y >= m * X + c), np.logical_not(tr), np.logical_not(bl)]), axis=0)
    br = 4. * np.all(np.array([(Y < m * X + c), np.logical_not(tr), np.logical_not(bl)]), axis=0)

    return bl + tr + tl + br


def perfect_virtual(x1, y1, x2, y2):

    point1 = np.array([
        [x1, y1]
    ])

    point2 = np.array([
        [x2, y2]
    ])

    x_amp = 1
    y_amp = 1
    res = 100

    x = np.linspace(-x_amp/2, x_amp/2, res)
    y = np.linspace(-y_amp/2, y_amp/2, res)

    X, Y = np.meshgrid(x, y)

    plane = np.stack(
        [X, Y], axis=0
    )

    plane2 = np.stack(
        [X, Y], axis=0
    )


    delta = (point2 - point1)[0]
    m = delta[1] / delta[0]
    c = point1[0][1] - m * point1[0][0]

    bl = 1. * np.all(plane < point1.T[..., np.newaxis], axis=0)
    tr = 2. * np.all(plane2 > point2.T[..., np.newaxis], axis=0)

    tl = 3. * np.all(np.array([(Y >= m * X + c), np.logical_not(tr), np.logical_not(bl)]), axis=0)
    br = 4. * np.all(np.array([(Y < m * X + c), np.logical_not(tr), np.logical_not(bl)]), axis=0)

    total = bl + tr + tl + br

    return total


point1 = np.array(
    [-0.1, -0.1]
)

point2 = np.array(
    [0.1, 0.1]
)

initial = (-0.1, -0.1, 0.1, 0.1, 0.1, 0., 0.1, 0.)

total = full(*point1, *point2, 0.1, -0.23, 0.05, 0.)

plt.figure()
plt.imshow(total.T, origin='lower', extent=(-0.5, 0.5, -0.5, 0.5))
plt.show()



from scipy.optimize import minimize

error = lambda params: np.sum((total - full(*params))**2)

# grad_error = grad(error)
plt.figure()
plt.imshow(full(*initial).T, origin='lower', extent=(-0.5, 0.5, -0.5, 0.5))

res = minimize(error, initial, method='Nelder-Mead')


# grad_error = grad(error)
plt.figure()
plt.imshow(full(*res.x).T, origin='lower', extent=(-0.5, 0.5, -0.5, 0.5))

