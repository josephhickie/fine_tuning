"""
Created on 25/08/2023
@author jdh
"""

from scipy.ndimage import gaussian_filter1d

def filter_trace(trace, sigma=3):
    return gaussian_filter1d(trace, sigma=sigma)
