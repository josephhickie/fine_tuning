"""
Created on 25/08/2023
@author jdh

Min-max scale array to values between 0 and 1.
"""

import numpy as np

def normalise(data):

    return (data - np.min(data)) / (np.max(data) - np.min(data))
