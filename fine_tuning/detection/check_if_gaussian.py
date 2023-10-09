"""
Created on 09/10/2023
@author jdh
"""

"""
Created on 30/08/2022
@author jdh
"""

from scipy.stats import anderson


def check_if_gaussian(samples, p_value=0.01):

    assert p_value in [0.15, 0.1, 0.05, 0.025, 0.01], "p value must be one of [0.15, 0.1, 0.05, 0.025, 0.01]"

    # selecting from fractional p value to array index for the anderson result
    p_values_dict = {0.15: 0, 0.1: 1, 0.05: 2, 0.025: 3, 0.01: 4}

    andy = anderson(samples, dist='norm')

    # returns true if anderson test statistic is greater than critical value for given significance level (p_value)
    # lower p value is more aggressive - default 0.01 is lowest available.
    return andy.statistic < andy.critical_values[p_values_dict.get(p_value)]