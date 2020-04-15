import os
from time import time
from datetime import timedelta

import h5py as h5
import numpy as np

from deepen import model

class CatClassifier:

    @staticmethod
    def run():
        classifier = CatClassifier()

        data_spec = {
            'datafile': 'data/data.h5',
            'x_name': 'data_x',
            'y_name': 'data_y',
            'class_name': 'data_class'
        }
        data_x, data_y, data_class = classifier.load_data(data_spec)

        norm_x, norm_y = classifier.normalize_data(data_x, data_y)
        train_x, train_y, test_x, test_y = classifier.partition_data(norm_x, norm_y)

        assert(train_x.shape == (12288, 209))
        assert(train_y.shape == (1, 209))
        assert(test_x.shape == (12288, 50))
        assert(test_y.shape == (1, 50))

        np.random.seed(1)
        layer_dims = [norm_x.shape[0], 20, 7, 5, 1]
        iterations = 2500

        print("Training model...", end='')
        start = time()
        params, costs = model.learn(train_x, train_y, layer_dims, iterations=iterations)
        end = time()
        print("done.")
        print("Time to train: %s" %  str(timedelta(seconds=(end - start))))

        for i in range(iterations):
            if i % 100 == 0:
                print("Cost at iteration %i: %f" % (i, costs[i]))

        train_pred = model.predict(train_x, params)
        train_accuracy = np.sum(train_pred == train_y) / train_y.shape[1]
        print("Training accuracy: %3.3f" % train_accuracy)

        test_pred = model.predict(test_x, params)
        test_accuracy = np.sum(test_pred == test_y) / test_y.shape[1]
        print("Test accuracy: %3.3f" % test_accuracy)

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
