import os

import h5py as h5
import numpy as np

from deepen.model import Model

class CatClassifier:

    def __init__(self):
        self.model = Model()

    @property
    def layer_dims(self):
        return self.model.layer_dims

    @layer_dims.setter
    def layer_dims(self, layer_dims):
        self.model.layer_dims = layer_dims

    def train(self, train_x, train_y, iterations):
        params, costs = self.model.learn(train_x, train_y, iterations=iterations)

        return (params, costs)

    def test(self, test_x):
        test_pred = self.model.predict(test_x)

        return test_pred

    def load_data(self, data_spec):
        """Load datasets from a specified H5-format file.

        Parameters
        ----------
        data_spec: dict of {str: str}
            Specifications for the data file.

            datafile: Path to the data file relative to this script.
            x_name: Name of the input dataset.
            y_name: Name of the output dataset.
            class_name: Name of the classifications dataset.

        Returns
        -------
        tuple of numpy array

            data_x: The input dataset.
            data_y: The output dataset.
            data_class: The classifications dataset.
        """

        path = os.path.dirname(os.path.abspath((__file__)))
        datapath = os.path.join(path, data_spec['datafile'])

        with h5.File(datapath, 'r') as datafile:
            data_x = datafile[data_spec['x_name']][:]
            data_y = datafile[data_spec['y_name']][:]
            data_class = datafile[data_spec['class_name']][:]

        return (data_x, data_y, data_class)

    def normalize_data(self, data_x, data_y):
        """Reshape and normalize the given datasets.

        Parameters
        ----------
        data_x: numpy array of int in [0..255]
            The input dataset, having shape (num_examples, x, y, 3).
        data_y: numpy array
            The output dataset, having shape (num_examples, 1).

        Returns
        -------
        tuple of numpy array

            norm_x: numpy array of float in [0..1]
                The input dataset, having shape(x * y * 3, num_examples).
            norm_y: numpy array
                The output dataset, having shape (1, num_examples).
        """

        norm_x = data_x.reshape(data_x.shape[0], -1).T
        norm_x = norm_x / 255

        norm_y = data_y.reshape(1, data_y.shape[0])

        return (norm_x, norm_y)

    def partition_data(self, data_x, data_y, test_frac=0.193):
        """Partition the data into training and test sets.

        Parameters
        ----------
        data_x: numpy array
            The input dataset, having shape (:, num_examples).
        data_y: numpy array
            The output dataset, having shape (1, num_examples).
        test_frac: float in [0..1]
            The fraction of the total number of examples in the dataset that
            should be reserved for the test set.

        Returns
        -------
        tuple of numpy array

            train_x: Training set input data, having shape (:, num_examples - test_examples).
            train_y: Training set output data, having shape (1, num_examples - test_examples).
            test_x: Test set input data, having shape (:, test_examples).
            test_y: Test set output data, having shape (1, test_examples).
        """

        test_cases = int(data_x.shape[1] * test_frac + 0.5)
        test_index = data_x.shape[1] - test_cases

        train_x = data_x[:, 0:test_index]
        train_y = data_y[:, 0:test_index]
        test_x = data_x[:, test_index:]
        test_y = data_y[:, test_index:]

        return (train_x, train_y, test_x, test_y)
