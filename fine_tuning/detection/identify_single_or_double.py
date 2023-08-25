"""
Created on 25/08/2023
@author jdh

For identifying whether a stability diagram measurement shows
a single line, in which case we assume we are not near a triple point,
or two lines of different orientations, in which case we are at a triple point
"""

from fine_tuning.helper_functions import normalise
from fine_tuning.helper_functions import flip_if_trough
from fine_tuning.helper_functions import filter_trace
from fine_tuning.data_handling import StabilityMeasurement

import numpy as np
from scipy.signal import find_peaks


def get_points(data, search_point_fraction=0.1):
    """
    Get the index at which to start the search traces.
    :param data: 2D array containing stability diagram measurement
    :param search_point_fraction: fraction into the measurement to place the start of the traces.
    :return: Array of [[x1, y1], [x2, y2]], index values for the first (top left) and second (bottom right) points.
    """

    x_lim, y_lim = data.shape

    points = (np.array([
        [search_point_fraction, 1 - search_point_fraction],
        [1 - search_point_fraction, search_point_fraction]
    ]) * np.array([x_lim - 1, y_lim - 1])).round().astype(int)

    return points


def get_traces(data, points):
    """
    Get all four traces from the data.
    :param data:
    :param points:
    :return:
    """

    p1 = points[0]
    p2 = points[1]

    p1_horizontal = data[p1[0]:, p1[1]]
    p1_vertical = data[p1[0], :p1[1]]

    p2_horizontal = data[:p2[0], p2[1]]
    p2_vertical = data[p2[0], p2[1]:]

    return [[p1_horizontal, p1_vertical], [p2_horizontal, p2_vertical]]

def peak_check_trace(trace, prominence_threshold_fraction=0.3):

    # to make sure we have a peak in positive direction
    trace = flip_if_trough(trace)

    # ensure data is in range [0, 1]
    trace = normalise(trace)

    # only look for peaks that have a relatively high prominence.
    peaks, properties = find_peaks(trace, prominence=prominence_threshold_fraction)

    return peaks, properties

def count_lines(data, search_point_fraction=0.1, filter_sigma=4, take_gradient=True):

    points = get_points(data, search_point_fraction)
    traces = get_traces(data, points)

    peaks_counted = []
    peak_idx = []

    for point_traces in traces:
        peak_idx_row = []
        for trace in point_traces:

            smoothed_trace = filter_trace(trace, sigma=filter_sigma)

            if take_gradient:
                smoothed_trace = np.gradient(smoothed_trace)

            peaks, properties = peak_check_trace(smoothed_trace)

            peaks_counted.append(len(peaks))

            peak_idx_row.append(peaks)

        peak_idx.append(peak_idx_row)

    return peaks_counted, peak_idx, traces


def identify(data, search_point_fraction=0.1, filter_sigma=4):

    peaks_counted, peak_idx, traces = count_lines(data, search_point_fraction, filter_sigma)
    peak_idx = np.array(peak_idx).flatten()

    measurement = StabilityMeasurement(
        data,
        peak_idx,
        traces,
        len(peak_idx)
    )

    return measurement

def plot_peaks(data, search_point_fraction=0.1, sigma=4):
    import matplotlib.pyplot as plt
    traces = get_traces(data, get_points(data, search_point_fraction))

    plt.figure()
    for point in traces:
        for trace in point:
            smoothed_trace = filter_trace(trace, sigma)
            trace_gradient = np.gradient(smoothed_trace)
            peaks, properties = peak_check_trace(trace_gradient)
            plt.plot(trace)
            plt.scatter(peaks, trace[peaks])

    plt.show()
