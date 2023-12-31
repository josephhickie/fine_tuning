"""
Created on 09/10/2023
@author jdh
"""

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import sys
sys.path.append('/home/jdh/PycharmProjects/qgor_qm')
from qgor import Rays
from qgor_simulation import Station
from qgor_simulation.Simulator import Simulator, SimulatorVideoMode
import numpy as np
from qgor import CompensatedDAC
from models import gmm_fit
from scipy.ndimage import sobel

from fine_tuning.detection import check_if_gaussian

station = Station()
dac = station.dummy_dac

simulator = Simulator(station)

from fine_tuning.detection import kde_detection
ray_generator = Rays([dac.dac3, dac.dac7])

measurement = lambda: simulator.do2d_fast()[0]


for dac_channel, value in zip(simulator.dac_channels, simulator.V0):
    dac_channel.set(value)


origin = dac.dict()

def reset():
    dac.set_from_dict(origin)


direction = [-1, -1]

def do1d_and_measure(ray, length, resolution, measure):
    results = []
    measurements = []
    for point in np.linspace(0, length, resolution):
        ray(point)
        data = measure()
        measurements.append(data)

        if check_if_gaussian(data.flatten()):
            detect = 0
            print('gaussian')

        # data = out.mm_r.get('PCA_0')
        else:
            detect = kde_detection(data)

        if detect == 3:
            kde_detection(data, plot=True)

        if detect == 2:
            vals = do_circle_check(*ray.parameters, measure)

            vals, counts = np.unique(vals, return_counts=True)
            if vals[np.argmax(counts)] == 3:
                kde_detection(data, plot=True)
                detect = 3

        results.append(detect)



    return np.array(results), np.array(measurements)
def do_circle_check(x_gate, y_gate, measure, x_range=20, y_range=20):

    print('checking area')

    x_origin = x_gate()
    y_origin = y_gate()

    results = []

    for x in np.linspace(-x_range / 2, x_range / 2, 5):
        x_gate(x_origin + x)
        for y in np.linspace(-y_range / 2, y_range / 2, 5):
            y_gate(y_origin + y)

            data = measure()

            detect = kde_detection(data)
            if detect == 3:
                kde_detection(data, plot=True)
            results.append(detect)

    return np.array(results)

if __name__ == '__main__':

    reset()
    r, m = do1d_and_measure(ray_generator.get_new_ray([-1, -1]), 100, 20, measurement)
    res, model, gm = gmm_fit(m[r == 3][0], plot=True)

    from fine_tuning.models.gmm.main import *
    e4 = lambda params: error(model.flatten(), do2d_(params).flatten())

    vals = minimize(e4, initial_params, method='Nelder-Mead')

    plt.figure()
    plt.imshow(m[r == 3][0].T, origin='lower')
    plt.figure()
    plt.imshow(do2d_(vals.x).T, origin='lower')



