from time import time
from datetime import timedelta

import numpy as np

import catclass

class CLI:

    @staticmethod
    def run():
        classifier = catclass.CatClassifier()

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
        params, costs = classifier.train(train_x, train_y, layer_dims, iterations=iterations)
        end = time()
        print("done.")
        print("Time to train: %s" %  str(timedelta(seconds=(end - start))))

        for i in range(iterations):
            if i % 100 == 0:
                print("Cost at iteration %i: %f" % (i, costs[i]))

        train_pred = classifier.test(train_x, params)
        train_accuracy = np.sum(train_pred == train_y) / train_y.shape[1]
        print("Training accuracy: %3.3f" % train_accuracy)

        test_pred = classifier.test(test_x, params)
        test_accuracy = np.sum(test_pred == test_y) / test_y.shape[1]
        print("Test accuracy: %3.3f" % test_accuracy)
