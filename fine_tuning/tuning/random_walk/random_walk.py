"""
Created on 14/09/2023
@author jdh
"""

import numpy as np
from pathlib import Path
import logging
import pickle

from fine_tuning.helper_functions import yaml_to_dict
from fine_tuning.helper_functions import create_directory_structure

from walker_dataclass import RandomWalkData

logger = logging.getLogger()

class RandomWalk:

    def __init__(self, station, awg_ramp_class, classification_fn, options_path=Path('./random_walk_options.yaml'), debug=False):
        self.station = station
        self.options = yaml_to_dict(options_path)
        self.awg_ramp = awg_ramp_class
        self.dac_limits = {}
        self.debug = debug
        self.classificaton_fn = classification_fn
        self.step_idx = 0

        self.num_dacs = len(self.options.get('dac_options'))
        self.jump_sizes = np.zeros(self.num_dacs)
        self.direction_probabilities = np.zeros(self.num_dacs)

        self.save_directory = Path(self.options.get('save_directory'))
        create_directory_structure(self.save_directory)

        self._setup()



    def _setup(self):

        dac_options = self.options.get('dac_options')
        for i, (dac_name, options) in enumerate(dac_options.items()):

            if self.debug:
                dac_parameter = dac_name
            else:
                dac_parameter = self.station.dac.__getattribute__(str(dac_name))

            self.dac_limits[dac_parameter] = np.array([options.get('min'), options.get('max')])

            self._dac_limit_array = np.array(list(self.dac_limits.values()))
            self.jump_sizes[i] = options.get('jump_size')
            self.direction_probabilities[i] = options.get('jump_up_probability')

    def _clip_new_values(self, new_values):

        clipped_new_values = np.clip(
            new_values, self._dac_limit_array[:, 0], self._dac_limit_array[:, 1]
        )

        return clipped_new_values

    def _generate_new_values(self):

        # rand or uniform ?
        direction = np.random.rand(self.num_dacs)
        direction = 2 * (direction > self.direction_probabilities).astype(int) - 1

        jumps = direction * self.jump_sizes

        if self.debug:
            current_dac_values = np.zeros(self.num_dacs)
        else:
            current_dac_values = np.array([dac() for dac in self.dac_limits.keys()])

        return jumps + current_dac_values

    def take_step(self):

        new_dac_values = self._clip_new_values(self._generate_new_values())


        for dac, new_value in zip(self.dac_limits.keys(), new_dac_values):
            if self.debug:
                print(f'debug: setting {dac} to {new_value}')
            else:
                dac(new_value)

        self.step_idx += 1

        data = self.measure()
        classify = 1
        dac_dict = {'a': np.random.rand()}

        dataclass = RandomWalkData(dac_dict, classify, data)
        self._save(dataclass, self.step_idx)




    def measure(self):
        # relies on the video mode plotter having been created

        if self.debug:
            return np.random.rand(62, 62)
        else:
            return self.awg_ramp._measure()

    def classify(self, data):
        return self.classificaton_fn(data.flatten())

    def _save(self, dataclass, step_idx):

        path = self.save_directory / (str(step_idx) + '.pickle')
        with open(path, 'wb') as handler:
            pickle.dump(dataclass, handler)



if __name__ == '__main__':

    walker = RandomWalk('station', 'ramp', lambda: 1, debug=True)




