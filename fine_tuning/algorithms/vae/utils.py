import numpy as np
import matplotlib.pyplot as plt
import time
import itertools
import numpy.random as npr
from tqdm.auto import trange
from os import listdir
from os.path import isfile, join
from pathlib import Path
import jax.numpy as jnp
import numpy as np
from numpy import load

from jax import device_put, jit, vmap
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

@jit
def normalise(data):
    return (data - data.min()) / (data.max() - data.min())


def data_stream(x, y, num_train, batch_size, num_batches):
    """
    A results stream for the training results
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
            x_b = device_put(x[batch_idx])
            y_b = device_put(y[batch_idx])
            yield x_b, y_b
def plot_examples(X, classify_fn, reconstruct_fn, title='Examples'):
    """
    Plot some examples of the results and the reconstruction
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
    plt.title(title)
    plt.tight_layout()
    plt.show()

# Load MNIST results
def load_data():
    """
    Load the MNIST results
    :return:
    """
    X, y = fetch_openml(
        "mnist_784", version=1, return_X_y=True, parser='liac-arff', as_frame=False, cache=False)

    a = X[0].reshape(28, 28)

    y_categorical = y.astype(int)
    label_binarizer = LabelBinarizer()
    label_binarizer.fit(range(max(y_categorical)+1))
    y_b = label_binarizer.transform(y_categorical)
    X = X/np.max(a)

    X_train, X_test, y_train, y_test = train_test_split(X, y_b, test_size=0.25, random_state=42)
    return X_train, X_test, y_train, y_test
def load_stability_data(**kwargs):

    X, y = fetch_dataset(**kwargs)
    # X_sim, y_sim = fetch_simulated_dataset()

    X = vmap(normalise, in_axes=0)(X)
    # X_sim = vmap(normalise, in_axes=0)(X_sim)
    #X = np.concatenate([X, X_sim], axis=0)
    #y = np.concatenate([y, y_sim], axis=0)

    def binarize(y):
        y_categorical = y.astype(int)
        label_binarizer = LabelBinarizer()
        label_binarizer.fit(range(max(y_categorical)+1))
        y_b = label_binarizer.transform(y_categorical)
        return y_b

    # y_sim_b = binarize(y_sim)
    y_b = binarize(y)

    X_train, X_test, y_train_b, y_test_b = train_test_split(X, y_b, test_size=0.25, random_state=42)

    X_train = np.concatenate([X_train])#, X_sim], axis=0)
    y_train_b = np.concatenate([y_train_b])#, y_sim_b], axis=0)

    return X_train, X_test, y_train_b, y_test_b

def fetch_simulated_dataset():
    directory = Path('/home/sebastiano/Documents/charge_stability_data/vae_training_data_simulated/triple')
    label = 3
    frame = []
    files = [f for f in listdir(directory) if isfile(join(directory, f))]
    for file in files:
        data = np.load(directory / file)
        data = data.flatten()
        # results = results[..., np.newaxis]
        frame.append(data)

    frame = np.array(frame)
    X, y = frame, np.ones(frame.shape[0]) * label

    return X, y
def fetch_dataset(root=None, folders=None, classes=None):
    root = '/home/jdh/Documents/vae_training/' if root is None else root

    classes = [0, 1, 2, 3] if classes is None else classes
    folders = ['noise', 'single_horizontal_with_compensation', 'single_vertical_with_compensation', 'triple_with_compensation'] if folders is None else folders

    complete_data = []
    complete_labels = []

    for data_class, folder in zip(classes, folders):
        frame = []
        directory = Path(root + folder)
        files = [f for f in listdir(directory) if isfile(join(directory, f))]

        for file in files:
            data = np.load(directory / file)
            data = normalise(data)
            # data = data.flatten()
            # results = results[..., np.newaxis]
            frame.append(data)

        frame = np.array(frame)
        labels = np.ones(frame.shape[0]) * data_class

        complete_data.append(frame)
        complete_labels.append(labels)

    X, y =  np.concatenate(complete_data, axis=0), np.concatenate(complete_labels, axis=0)

    return X, y







