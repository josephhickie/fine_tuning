"""
Created on 12/10/2023
@author jdh
"""

import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from fine_tuning.algorithms.vae.utils import fetch_dataset
from torch.utils.data import TensorDataset, DataLoader

X_, y = fetch_dataset()

# x to false RGB
X = np.concatenate([X_[..., np.newaxis]] * 3, axis=-1)


