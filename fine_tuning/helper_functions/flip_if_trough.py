"""
Created on 25/08/2023
@author jdh

If the mean is less than the median, there is
something dragging the mean down. This is most likely a trough
rather than a peak. Returns the negative of the trough to
give us a peak instead.
"""

import numpy as np

def flip_if_trough(trace):

    if np.mean(trace) > np.median(trace):
        return trace

    else:
        return - trace
