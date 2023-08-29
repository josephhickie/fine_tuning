"""
Created on 28/08/2023
@author jdh
"""

# import the modules used in this program:
import sys, os, time
import qcodes
import numpy as np

import matplotlib.pyplot as plt
# %matplotlib inline

import scipy.optimize
import cv2
import qtt
import qtt.measurements
from qtt.algorithms.anticrossing import fit_anticrossing, plot_anticrossing
from qtt.data import load_example_dataset


data = load_example_dataset('charge_stability_diagram_anti_crossing')

qtt.data.plot_dataset(data, fig=10)
fit_results = fit_anticrossing(data)
plot_anticrossing(data, fit_results)



from qgor_simulation import Station


station = Station()
data = station.database.load(4)
d = data.mm_r.get('PCA_0')
da = station.database.load(5).mm_r.get('PCA_0')

dat = np.mean(np.gradient(da, axis=(0, 1)), axis=0)
uh = dat[4: ]


