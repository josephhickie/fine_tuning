"""
Created on 25/08/2023
@author jdh

Finds an approximate centre of triple point.
"""


from .identify_single_or_double import count_lines
import numpy as np

def get_triple_point_idx(data, search_point_fraction=0.1):

    peaks_counted, idx, traces = count_lines(data, search_point_fraction)
    idx = np.array(idx).flatten()

    mean_vertical = np.mean([idx[0], idx[2]])
    mean_horizontal = np.mean([idx[1], idx[3]])

    return np.array([mean_horizontal, mean_vertical])
