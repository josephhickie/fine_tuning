"""
Created on 12/10/2023
@author jdh
"""

import flax.linen as nn
import jax.nn

NUM_CLASSES = 120  # We are told this by the Stanford Dogs dataset website.

import torchvision

all_images = torchvision.datasets.ImageFolder("Images/")


class VGG19(nn.Module):
    @nn.compact
    def __call__(self, x, training):
        x = self._stack(x, 64, training)
        x = self._stack(x, 64, training)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        x = self._stack(x, 128, training)
        x = self._stack(x, 128, training)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        x = self._stack(x, 256, training)
        x = self._stack(x, 256, training)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        x = self._stack(x, 512, training)
        x = self._stack(x, 512, training)
        x = self._stack(x, 512, training)
        x = self._stack(x, 512, training)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        x = self._stack(x, 512, training)
        x = self._stack(x, 512, training)
        x = self._stack(x, 512, training)
        x = self._stack(x, 512, training)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        x = x.reshape((x.shape[0], -1))

        x = nn.Dense(features=4096)(x)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.relu(x)
        x = nn.Dropout(0.5, deterministic=not training)(x)

        x = nn.Dense(features=4096)(x)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.relu(x)
        x = nn.Dropout(0.5, deterministic=not training)(x)

        x = nn.Dense(features=NUM_CLASSES)(x)
        return x

    @staticmethod
    def _stack(x, features, training, dropout=None):
        x = nn.Conv(features=features, kernel_size=(3, 3), padding='SAME')(x)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.relu(x)
        return x


import os.path
import xml.etree.ElementTree as ET
from torchvision import transforms
from PIL import Image


def get_bounding_box(image_path):
    """Gets the bounding box for the dog in the image at image_path."""
    (_, label, filename) = image_path.split('/')
    (name, _) = os.path.splitext(filename)
    tree = ET.parse(f'Annotation/{label}/{name}')
    o = tree.getroot().findall('object')[0]
    bndbox = o.find('bndbox')
    xmin = int(bndbox.find('xmin').text)
    ymin = int(bndbox.find('ymin').text)
    xmax = int(bndbox.find('xmax').text)
    ymax = int(bndbox.find('ymax').text)
    return (xmin, ymin, xmax, ymax)


image_cache = {}


def cropped_loader(image_path):
    global image_cache
    if not (image_path in image_cache):
        box = get_bounding_box(image_path)
        image_cache[image_path] = Image.open(image_path).convert('RGB').crop(box)
    return image_cache[image_path]


all_images = torchvision.datasets.ImageFolder("Images/", loader=cropped_loader)

import torch
from copy import copy


def split_dataset(all_images):
    num_images = len(all_images)
    num_training_images = int(0.8 * num_images)
    num_testing_images = num_images - num_training_images
    return torch.utils.data.random_split(all_images,
                                         [num_training_images,
                                          num_testing_images])


def apply_to_image(f):
    def inner(image_and_label):
        return (f(image_and_label[0]), image_and_label[1])

    return inner


torch.manual_seed(42)

IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224

training_transform = transforms.Compose([
    transforms.RandomAffine(degrees=(-30, 30),
                            translate=(0.0, 0.2)),
    transforms.RandomHorizontalFlip(),
    transforms.Resize((IMAGE_HEIGHT,
                       IMAGE_WIDTH)),
    np.array])

testing_transform = transforms.Compose([
    transforms.Resize((IMAGE_HEIGHT,
                       IMAGE_WIDTH)),
    np.array])

train_dataset, test_dataset = split_dataset(all_images)

# Unfortunately torch datasets can't be map()ed over while retaining __len__,
# which DataLoaders (next up) need, so we copy the whole dataset, and change its
# .transform as needed.
train_dataset.dataset = copy(all_images)
train_dataset.dataset.transform = training_transform
test_dataset.dataset.transform = testing_transform

import jax

NUM_TPUS = jax.device_count()
BATCH_SIZE = 64
train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=NUM_TPUS * BATCH_SIZE,
                                               shuffle=True, drop_last=True,
                                               num_workers=2)
test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=NUM_TPUS * BATCH_SIZE,
                                              shuffle=True, drop_last=True,
                                              num_workers=2)

(image_batch, label_batch) = next(iter(train_dataloader))
print(image_batch.shape)
print(label_batch.shape)

