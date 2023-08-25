"""
Created on 25/08/2023
@author jdh
"""

from dataclasses import dataclass
import numpy as np

@dataclass
class StabilityMeasurement:

    data: np.ndarray
    peaks_idx: np.ndarray
    traces: list
    num_peaks: int

    def __post_init__(self):

        if np.all(np.array(self.peaks_idx) == 1):
            self.classification = 2
        elif np.all(np.sort(self.peaks_idx) == np.array([0, 0, 1, 1])):
            self.classification = 1
        else:
            self.classification = 0



