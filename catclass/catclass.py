import os

import h5py as h5

class CatClassifier:

    @staticmethod
    def run():
        classifier = CatClassifier()

        data_x, data_y, data_class = classifier.load_data()

        norm_x, norm_y = classifier.normalize_data(data_x, data_y)
        train_x, train_y, test_x, test_y = classifier.partition_data(norm_x, norm_y)

        assert(train_x.shape == (12288, 209))
        assert(train_y.shape == (1, 209))
        assert(test_x.shape == (12288, 50))
        assert(test_y.shape == (1, 50))

        print("I am not a cat!")

    def load_data(self):
        path = os.path.dirname(os.path.abspath((__file__)))
        datapath = os.path.join(path, 'data/data.h5')

        with h5.File(datapath, 'r') as datafile:
            data_x = datafile['data_x'][:]
            data_y = datafile['data_y'][:]
            data_class = datafile['data_class'][:]

        return (data_x, data_y, data_class)

    def normalize_data(self, data_x, data_y):
        norm_x = data_x.reshape(data_x.shape[0], -1).T
        norm_x = norm_x / 255

        norm_y = data_y.reshape(1, data_y.shape[0])

        return (norm_x, norm_y)

    def partition_data(self, data_x, data_y, test_frac=0.193):
        test_cases = int(data_x.shape[1] * test_frac + 0.5)
        test_index = data_x.shape[1] - test_cases

        train_x = data_x[:, 0:test_index]
        train_y = data_y[:, 0:test_index]
        test_x = data_x[:, test_index:]
        test_y = data_y[:, test_index:]

        return (train_x, train_y, test_x, test_y)
