"""
Created on 14/09/2023
@author jdh
"""

import yaml

def yaml_to_dict(yaml_file_path):
    with open(yaml_file_path, 'r') as file:
        return yaml.safe_load(file)

