"""
Created on 14/09/2023
@author jdh
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class RandomWalkData:
    dac_dict: dict
    classification: int
    data: np.ndarray