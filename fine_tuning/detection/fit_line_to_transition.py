"""
Created on 25/08/2023
@author jdh
"""

from fine_tuning.data_handling import Line

from scipy.stats import linregress
import numpy as np
from scipy.ndimage import gaussian_filter



def _fit_line(x, y):
    m, c, r, p_value, std_error = linregress(x, y)
    line = Line(m=m, c=c, r=r, p_value=p_value, std_error=std_error, x=x, y=y)

    return line


# filter and gradient data first, remove edges
def fit_single_line(filtered_data, edge_safety=10, line_to_data_min_ratio=1):
    max_X = np.stack([np.arange(0, filtered_data.shape[1]), filtered_data.argmax(axis=0)], axis=0)

    max_X = max_X[
            :, np.logical_and(
        max_X[1, :] > edge_safety,
        max_X[1, :] < filtered_data.shape[1] - edge_safety
    )
            ]

    line_x = _fit_line(*max_X)

    return line_x


def fit_separately(filtered_data, edge_safety=10, line_to_data_min_ratio=1):

    # get max values in X direction and in Y direction as two different sets so we can see
    # which one fits better
    max_X = np.stack([np.arange(0, filtered_data.shape[1]), filtered_data.argmax(axis=0)], axis=0)
    max_Y = np.stack([filtered_data.argmax(axis=0), np.arange(0, filtered_data.shape[1])], axis=0)


    max_X = max_X[
            :, np.logical_and(
        max_X[1, :] > edge_safety,
        max_X[1, :] < filtered_data.shape[1] - edge_safety
    )
            ]

    max_Y = max_Y[
            :, np.logical_and(
        max_Y[0, :] > edge_safety,
        max_Y[0, :] < filtered_data.shape[0] - edge_safety
    )
            ]


    # if these do not have a size, they are empty -> no peak is found
    if max_X.size != 0:
        line_x = _fit_line(*max_X)
    if max_Y.size != 0:
        line_y = _fit_line(*max_Y)

    line = line_x if line_x.r ** 2 > line_y.r ** 2 else line_y

    return line
