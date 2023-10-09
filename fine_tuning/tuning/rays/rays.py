"""
Created on 09/10/2023
@author jdh
"""

from fine_tuning.helper_functions import yaml_to_dict

from pathlib import Path
from random import gauss

sys.path.append('/home/jdh/PycharmProjects/qgor_qm')
# from qgor_simulation import Station
from qgor import ArbitraryOriginVector

class Rays:

    def __init__(self, station, dacs, options_path=Path('./rays.yaml')):
        self.options = yaml_to_dict(options_path)
        self.station = station
        self.dacs = dacs

    def get_new_ray(self):
        vec = [gauss(0, 1) for i in range(len(self.dacs))]
        mag = sum(x ** 2 for x in vec) ** .5
        direction = [x / mag for x in vec]
        return ArbitraryOriginVector(
            parameters=self.dacs,
            direction=direction
        )




if __name__=='__main__':
    ray = Rays('a', ['dac1', 'dac2', 'dac3'])

