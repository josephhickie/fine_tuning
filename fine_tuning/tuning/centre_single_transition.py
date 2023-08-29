"""
Created on 25/08/2023
@author jdh
"""

import numpy as np
from fine_tuning.detection import count_lines


def flatten(matrix):
    vals = [item for row in matrix for item in row]
    for i, val in enumerate(vals):
        if np.size(val) == 0:
            vals[i] = np.nan
        else:
            vals[i] = val[0]

    return np.array(vals)

def identify_vertical_or_horizontal(peaks_counted):

    # vertical traces have peaks therefore horizontal peak
    if np.all(np.array(peaks_counted).astype(bool) == [False, True, False, True]):
        return 0

    # horizontal traces have peaks therefore vertical peak
    elif np.all(np.array(peaks_counted).astype(bool) == [True, False, True, False]):
        return 1

    else:
        print(peaks_counted)
        print('something weird is going on. More than one peak here.')
        return None

def centre_single_transition(data, size_x, size_y, take_gradient=True):

    shape = np.array(data.shape)

    centre_point_idx = shape / 2

    # yes i know it's inefficient, i am just making it work
    peaks_counted, peak_idx, traces = count_lines(data, search_point_fraction=0.1, filter_sigma=4, take_gradient=take_gradient)
    coulomb_peak_direction = identify_vertical_or_horizontal(peaks_counted)

    peak_idx = flatten(peak_idx)

    # horizontal coulomb peak therefore need to move y to centre
    if coulomb_peak_direction == 0:
        print('horizontal')

        y_size = data.shape[1]
        y_current = np.nanmean(peak_idx)
        y_centre = y_size / 2

        difference = (y_current - y_centre) / y_size

        return 1, difference * size_y


    # coulomb peak vertically, therefore need to move x to centre
    elif coulomb_peak_direction == 1:
        print('vertical')

        x_size = data.shape[0]
        x_current = np.nanmean(peak_idx)
        x_centre = x_size / 2

        difference = (x_current - x_centre) / x_size

        return 0, difference * size_x

    else:
        return None













