"""
Created on 25/08/2023
@author jdh
"""

from fine_tuning.detection import get_triple_point_idx
import numpy as np

def centre_triple_point_param_space(data, size_x, size_y):
    """
    gives the required movement in parameter x and parameter y to centre the triple point found in data.
    :param stability_measurement:
    :param size_x:
    :param size_y:
    :return:
    """

    # Assumes we are already at a triple point as identified by one of the other methods.
    shape = np.array(data.shape)
    centre_point_idx = shape / 2
    triple_point_idx = get_triple_point_idx(data)

    difference = centre_point_idx - triple_point_idx
    mv_per_pixel = np.array([size_x, size_y]) / shape

    mv_change = difference * mv_per_pixel

    return mv_change
