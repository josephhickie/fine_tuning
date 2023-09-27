"""
Created on 25/09/2023
@author jdh
"""

import numpy as np
from itertools import product

def constant_interaction(Cdd_inv, Cgd, Vg, V0, maximum_number_of_holes, **kwargs):
    Vg = V0[np.newaxis, np.newaxis, :] - Vg

    N_continuous = np.einsum('ij, abj', Cgd, Vg)

    # all combinations with replacement of floor and ceiling operations for each dot to work out the occupation
    floor_ceiling_list = product([np.floor, np.ceil], repeat=N_continuous.shape[2])

    # whomever understands this line of code, should he be worthy, will get a PhD :)
    N_discrete = np.stack([np.stack([operator(data)
                                     for operator, data
                                     in zip(operations, np.rollaxis(N_continuous, 2))], axis=2)
                           for operations
                           in floor_ceiling_list], axis=0)

    N_discrete = np.clip(N_discrete, a_min=0, a_max=maximum_number_of_holes)
    v = N_discrete - N_continuous
    U = np.einsum('abcd, de, abce -> abc', v, Cdd_inv, v)

    return N_continuous, N_discrete, U



cdd =  np.array([
      [1, -0.2, -0.01],
      [-0.2, 1, -0.01],
      [-0.01, -0.01, 1]
    ])

cdg = np.array([
      [0.041, 0.043, 0.04, 0.001, 0.001, 0.01, 0.001, 0.001],
      [0.01, 0.002, 0.038, 0.027, 0.03, 0.001, 0.001, 0.001],
      [0.0001, 0.0002, 0.0001, 0.0001, 0.0001, 0.03, 0.02, 0.03]
    ])
cpg = np.array([
      [0.1, 0.01, 0.04, 0, 0, 0, 0, 0],
      [0.02, 0.01, 0.06, 0.01, 0.02, 0.00, 0.00, 0.00],
      [0, 0, 0.01, 0.01, 0.1, 0.00, 0.00, 0.00]
    ])

cdd_inv = np.linalg.inv(cdd)

pinch_off_matrix = np.stack([
            cpg[0, 0:5],
            cdg[0, 0:5],
            cpg[1, 0:5],
            cdg[1, 0:5],
            cpg[2, 0:5]
        ], axis=0)

pinch_off_points = np.array([100, 100, 100, 100, 100])

V0 = np.linalg.solve(pinch_off_matrix, pinch_off_points)
V0 = np.concatenate([V0, np.array([1000, 1000, 1000])])
maximum_number_of_holes = 5

x = lambda vg: constant_interaction(cdd_inv, cdg, vg, V0, maximum_number_of_holes)

fast_amplitude = [10, 10]
fast_resolution = [20, 20]

Z = np.meshgrid(
            np.linspace(-fast_amplitude[0] / 2, fast_amplitude[0] / 2, fast_resolution[0]),
            np.linspace(-fast_amplitude[1] / 2, fast_amplitude[1] / 2, fast_resolution[1]),
        )

Vg = []
for dac_channel in self.simulator.dac_channels:
    if dac_channel.name in self.simulator.fast_channels:
        for voltages, fast_channel in zip(Z, self.simulator.fast_channels):
            if dac_channel.name == fast_channel:
                Vg.append(
                    voltages + dac_channel.cache()
                )

    else:
        Vg.append(
            np.full(shape=self.simulator.fast_resolution, fill_value=dac_channel.cache())
        )
Vg = np.stack(Vg, axis=2)
