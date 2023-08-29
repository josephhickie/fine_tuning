"""
Created on 25/08/2023
@author jdh
"""

# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt

import sys
sys.path.append('/home/jdh/PycharmProjects/qgor_qm')

from fine_tuning.detection import identify
from fine_tuning.detection import plot_peaks

from fine_tuning.tuning import centre_triple_point_param_space
from fine_tuning.detection import get_triple_point_idx
from fine_tuning.detection import fit_separately
from fine_tuning.detection import fit_single_line
from fine_tuning.tuning import centre_single_transition
from fine_tuning.tuning import follow_single_transition

from qgor_simulation import Station
import numpy as np


station = Station()
data = station.database.load(1)
triple = data.mm_r.get('D')


# d = data.mm_r.get('PCA_0')
da = station.database.load(5).mm_r.get('PCA_0')

dat = np.mean(np.gradient(da, axis=(0, 1)), axis=0)
uh = dat[4: ]




import scipy.optimize
import cv2
import qtt
import qtt.measurements
from qtt.algorithms.anticrossing import fit_anticrossing, plot_anticrossing
from qtt.data import load_example_dataset
import time
from qtt.algorithms.images import straightenImage
from qtt.utilities.imagetools import cleanSensingImage
from qtt.utilities.tools import showImage as showIm
from qtt.measurements.scans import fixReversal
from qtt.utilities.imagetools import fitModel, evaluateCross

rot_data = np.rot90(triple)

# imextent = (0, 10, 0, 10)
# istep = 0.25
# imc = cleanSensingImage(rot_data, sigma=0.93, verbose=1)
#
# imx, (fw, fh, mvx, mvy, Hstraight) = straightenImage(imc, imextent, mvx=istep, verbose=2)
#
# imx = imx.astype(np.float64)*(100./np.percentile(imx, 99)) # scale image
#
# showIm(imx, fig=100, title='straight image')
#
# istepmodel = .5
# ksizemv = 31
# param0 = [(imx.shape[0] / 2 + .5) * istep, (imx.shape[0] / 2 + .5) * istep, \
#           3.5, 1.17809725, 3.5, 4.3196899, 0.39269908]
# param0e = np.hstack((param0, [np.pi / 4]))
# cost, patch, r, _ = evaluateCross(param0e, imx, verbose=0, fig=21, istep=istep, istepmodel=istepmodel)
#
# t0 = time.time()
# res = qtt.utilities.imagetools.fitModel(param0e, imx, verbose=1, cfig=10, istep=istep,
#                    istepmodel=istepmodel, ksizemv=ksizemv, use_abs=True)
# param = res.x
# dt = time.time() - t0
# print('calculation time: %.2f [s]' % dt)
#
# cost, patch, cdata, _ = evaluateCross(param, imx, verbose=1, fig=25, istep=istep, istepmodel=istepmodel, linewidth=4)
