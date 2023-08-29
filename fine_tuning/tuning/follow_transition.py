"""
Created on 25/08/2023
@author jdh
"""

import numpy as np
from fine_tuning.detection import fit_separately

def follow_single_transition(data, delta):

    line = fit_separately(data)

    if np.abs(line.m) >= 1:

        delta_y = delta
        delta_x = (delta_y - line.c) / line.m

        return delta_x, delta_y

    else:

        delta_x = delta
        delta_y = (line.m * delta_x) + line.c

        return delta_x, delta_y




