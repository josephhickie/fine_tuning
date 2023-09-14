"""
Created on 25/08/2023
@author jdh

Standardise results (mean = 0, variance = 1)
"""

import numpy as np

def standardise(data):

    return (data - np.mean(data)) / np.var(data)
