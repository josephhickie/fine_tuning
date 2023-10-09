"""
Created on 09/10/2023
@author jdh
"""
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import argrelextrema
from sklearn.neighbors import KernelDensity

def normalise(data):
    return (data - data.min()) / (data.max() - data.min())

def kde_detection(data, scale=35, score_sample_rate=3, plot=False, **kwargs):
    """
    :param data:
    :param scale:
    :param score_sample_rate:
    :param plot:
    :param kwargs:
    :return:
    """

    data = normalise(data)
    data = data * scale
    kde = KernelDensity(kernel='gaussian', **kwargs).fit(data.reshape(-1, 1))

    s = np.linspace(data.min(), data.max(), scale * score_sample_rate)
    e = kde.score_samples(s.reshape(-1, 1))
    mi, ma = argrelextrema(e, np.less)[0], argrelextrema(e, np.greater)[0]

    if plot:
        plt.figure()
        plt.imshow(data.reshape(62, 62).T, origin='lower')
        plt.show()

        plt.figure()
        plt.plot(
            # s[:mi[0] + 1], e[:mi[0] + 1], 'r',
            #      s[mi[0]:mi[1] + 1], e[mi[0]:mi[1] + 1], 'g',
            #      s[mi[1]:mi[2] + 1], e[mi[1]:mi[2] + 1], 'b',
            #      s[mi[2]:], e[mi[2]:], 'k',
            s[ma], e[ma], 'go',
            s[mi], e[mi], 'ro')
        plt.plot(s, e)

    if len(mi) == 3 and len(ma) == 4:
        # print('four states')
        return 3
    elif len(mi) == 2 and len(ma) == 3:
        # print('three states')
        return 2
    elif len(mi) == 1 and len(ma) == 2:
        # print('two states')
        return 1
    else:
        return 0
        # print('one state / noise')