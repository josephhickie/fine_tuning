"""
Created on 13/10/2023
@author jdh
"""
import matplotlib

matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import torch
from torchvision import models, transforms
import torch.nn as nn
import numpy as np

from fine_tuning.algorithms.vae.utils import fetch_dataset

gpu = 1
device = f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu'

model_dir = './saved_models/resnet50_pretrained_15k_50ep.pt'
model = models.resnet50(pretrained=True)
model = model.to(device)


def normalise(data):
    return (data - data.min()) / (data.max() - data.min())


num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 4).to(device)

model_dict = torch.load(model_dir, map_location=torch.device('cpu'))

model.load_state_dict(model_dict)
reshape_size = (224, 224)
resize = transforms.Resize(size=reshape_size)

X, y = fetch_dataset(root='/home/jdh/Documents/vae_training/',
                     classes=[0, 1, 2, 3],
                     folders=['noise', 'single_horizontal_with_compensation', 'single_vertical_with_compensation',
                              'triple_with_compensation']
                     )

reshape_size = (224, 224)
resize = transforms.Resize(size=reshape_size)

X = np.concatenate([X[..., np.newaxis]] * 3, axis=-1)
X = np.rollaxis(X, -1, 1)
tensor_x = torch.Tensor(X).to(device)  # transform to torch tensor
tensor_y = torch.Tensor(y).type(torch.LongTensor).to(device)
tensor_x = resize(tensor_x)


def infer(data):
    data = normalise(data)
    data = np.concatenate([data[..., None]] * 3, axis=-1)
    data = np.rollaxis(data, -1, 0)
    data = torch.Tensor(data)
    data = resize(data)
    data = data[None, ...]
    with torch.no_grad():
        x = model(data)

    return x
