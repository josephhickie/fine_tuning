import numpy as np
import matplotlib.pyplot as plt
import time
import itertools
import numpy.random as npr
from tqdm.auto import trange
from os import listdir
from os.path import isfile, join
from pathlib import Path
import jax.numpy as np
from numpy import load

import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

def normalise(data):

    return (data - data.min()) / (data.max() - data.min())


def data_stream(X_train, y_train, num_train, batch_size, num_batches):
    """
    A data stream for the training data
    :param X_train:
    :param y_train:
    :param num_train:
    :param batch_size:
    :param num_batches:
    :return:
    """
    rng = npr.RandomState(0)
    while True:
        perm = rng.permutation(num_train)
        for i in range(num_batches):
            batch_idx = perm[i * batch_size:(i + 1) * batch_size]
            yield X_train[batch_idx], y_train[batch_idx]
def plot_examples(X, classify_fn, reconstruct_fn):
    """
    Plot some examples of the data and the reconstruction
    :param X:
    :param classify_fn:
    :param reconstruct_fn:
    :return:
    """
    fig, axs = plt.subplots(2, 4)

    # generate random integers in range 0 to len(X)
    indxs = npr.randint(0, len(X), 4)
    for i, item in enumerate(indxs):
        axs[0, i].imshow(X[item].reshape((62, 62)))
        axs[1, i].imshow(reconstruct_fn(X[item]).reshape((62, 62)))
        predict_classs = classify_fn(X[item])
        axs[0, i].set_xticks([])
        axs[0, i].set_yticks([])
        axs[1, i].set_xticks([])
        axs[1, i].set_yticks([])
        axs[0, i].set_title("Class: {}".format(predict_classs))

    axs[0, 0].set_ylabel("true")
    axs[1, 0].set_ylabel("reconstructed")
    plt.show()

# Load MNIST data
def load_data():
    """
    Load the MNIST data
    :return:
    """
    X, y = fetch_openml(
        "mnist_784", version=1, return_X_y=True, as_frame=False, cache=False)

    a = X[0].reshape(28, 28)

    y_categorical = y.astype(int)
    label_binarizer = LabelBinarizer()
    label_binarizer.fit(range(max(y_categorical)+1))
    y_b = label_binarizer.transform(y_categorical)
    X = X/np.max(a)

    X_train, X_test, y_train, y_test = train_test_split(X, y_b, test_size=0.25, random_state=42)
    return X_train, X_test, y_train, y_test

def load_stability_data():

    X, y = fetch_dataset()
    a = X[0].reshape(62, 62)

    y_categorical = y.astype(int)
    label_binarizer = LabelBinarizer()
    label_binarizer.fit(range(max(y_categorical)+1))
    y_b = label_binarizer.transform(y_categorical)
    X = X/np.max(a)

    X_train, X_test, y_train, y_test = train_test_split(X, y_b, test_size=0.25, random_state=42)
    return X_train, X_test, y_train, y_test


def fetch_dataset():
    root = '/home/jdh/Documents/vae_training/'

    classes = [0, 1, 2, 3]
    folders = ['noise', 'single_horizontal_with_compensation', 'single_vertical_with_compensation', 'triple_with_compensation']

    complete_data = []
    complete_labels = []

    for data_class, folder in zip(classes, folders):
        frame = []
        directory = Path(root + folder)
        files = [f for f in listdir(directory) if isfile(join(directory, f))]

        for file in files:
            data = np.load(directory / file)
            data = normalise(data)
            data = data.flatten()
            # data = data[..., np.newaxis]
            frame.append(data)

        frame = np.array(frame)
        labels = np.ones(frame.shape[0]) * data_class

        complete_data.append(frame)
        complete_labels.append(labels)

    X, y =  np.concatenate(complete_data, axis=0), np.concatenate(complete_labels, axis=0)

    return X, y







