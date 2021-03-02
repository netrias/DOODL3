import os
import random
import numpy as np
import pandas as pd
import keras
from tensorflow.keras.datasets import mnist
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def _remove_numbers(x, y, list_of_numbers_to_keep):
    y_df = pd.DataFrame(y, columns=['Label'])
    y_df = y_df[y_df['Label'].apply(str).isin(list_of_numbers_to_keep)]
    x = x[y_df.index, :, :, :]
    y = y_df.to_numpy()
    return x, y.flatten()


def load_mnist_data(subset_the_test_set=True):
    """
    :param subset_the_test_set: if True, will return a random subset of the mnist data
    :return:
    """
    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(60000, 28, 28, 1)
    x_test = x_test.reshape(10000, 28, 28, 1)
    nums_to_keep = ['0', '1']
    num_classes = len(nums_to_keep)
    x_train, y_train = _remove_numbers(x_train, y_train, nums_to_keep)
    x_test, y_test = _remove_numbers(x_test, y_test, nums_to_keep)

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    if subset_the_test_set:
        random.seed(5)
        rand_numbers = random.sample(range(0, 1000), 100)
        x_test = x_test[rand_numbers, :, :, :]
        y_test = y_test[rand_numbers, :]

    return x_train, y_train, x_test, y_test


def retrieve_predictions(y_test, outdir) -> pd.DataFrame:
    """
    Retrieves and cleans up predictions that were generated from using train_model_iteratively.

    :param y_test: should be the same as what was passed-in to Y_test in train_model_iteratively
    :param outdir: should be the same as what was passed-in to outdir in train_model_iteratively
    :return: DataFrame of cleaned predictions.
    """
    label_cols = list(range(y_test.shape[1]))
    yDF = pd.DataFrame(y_test, columns=label_cols)
    yDF['point'] = range(len(y_test))
    y_pred = pd.read_csv(os.path.join(outdir, 'predictions.csv'))
    y_pred.rename({y_pred.columns[0]: 'point'}, axis=1, inplace=True)
    y_pred = pd.merge(yDF, y_pred, left_on='point', right_on='point')

    y_pred["class"] = y_pred["class"].str.rstrip("',)'").str.lstrip("('p").astype(int)
    y_pred = y_pred.rename(columns={"class": "pred_class"})
    y_pred["real_class"] = y_pred[label_cols].idxmax(axis=1)
    y_pred = y_pred.drop(columns=label_cols)

    return y_pred


def load_iris_data(subset_train_features=True):
    iris = load_iris()
    x = iris['data']
    y = iris['target']
    names = iris['target_names']
    feature_names = iris['feature_names']

    if subset_train_features:
        x = x[:, [2, 3]]

    # One hot encoding
    enc = OneHotEncoder()
    y = enc.fit_transform(y[:, np.newaxis]).toarray()

    # Scale data to have mean 0 and variance 1 (important for convergence of the neural network)
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    # Split the data set into training and testing
    x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.5, random_state=2)
    return x_train, x_test, y_train, y_test
