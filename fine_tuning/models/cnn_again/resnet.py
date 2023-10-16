"""
Created on 12/10/2023
@author jdh
"""

# Import Libraries
import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from typing import Callable
from tqdm.notebook import tqdm


from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
import tensorflow as tf
import tensorflow_datasets as tfds 

import jax
import optax
import flax
import jax.numpy as jnp
from jax import jit
from jax import lax
from jax_resnet import pretrained_resnet, slice_variables, Sequential
from flax.jax_utils import replicate, unreplicate
from flax.training import train_state
from flax import linen as nn
from flax.core import FrozenDict,frozen_dict
from flax.training.common_utils import shard

import warnings
import logging
from functools import partial

warnings.simplefilter('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 