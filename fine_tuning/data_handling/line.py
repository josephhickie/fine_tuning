"""
Created on 25/08/2023
@author jdh
"""

class Line:

    def __init__(self, **kwargs):

        for key, value in kwargs.items():
            self.__setattr__(key, value)

    def __repr__(self):
        string = 'm: {}\n' \
                 'c: {}\n' \
                 f'r_squared: {self.r**2}'

        return string
