"""
Created on 25/08/2023
@author jdh
"""

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import sys
sys.path.append('/home/jdh/PycharmProjects/qgor_qm')

from fine_tuning.detection import identify
from fine_tuning.detection import plot_peaks

from fine_tuning.tuning import centre_triple_point_param_space
from fine_tuning.detection import get_triple_point_idx
from fine_tuning.detection import fit_separately
from fine_tuning.detection import fit_single_line

from qgor_simulation import Station
import numpy as np


station = Station()
data = station.database.load(4)
d = data.mm_r.get('PCA_0')
da = station.database.load(5).mm_r.get('PCA_0')

dat = np.mean(np.gradient(da, axis=(0, 1)), axis=0)
uh = dat[4: ]


