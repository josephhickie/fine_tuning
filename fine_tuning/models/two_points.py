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

initial = (-0.248, -0.34, 0.18097, 0.1)

total = perfect_virtual(*point1, *point2)

plt.figure()
plt.imshow(total.T, origin='lower')
plt.show()

t2 = perfect_virtual(*initial)

plt.figure()
plt.imshow(t2.T, origin='lower')
plt.show()


from scipy.optimize import minimize

error = lambda params: np.sum((total - perfect_virtual(*params))**2)

# grad_error = grad(error)



skew = tf.AffineTransform(shear=(2, 2))

mod = tf.warp(total, inverse_map=skew)

plt.figure()
plt.imshow(mod.T, origin='lower')
plt.show()