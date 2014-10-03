from pylearn2.datasets.cifar10 import CIFAR10
from pylearn2.space import Conv2DSpace

import theano
from theano import tensor as T
import numpy as np
from utils import *


class MCDNN():

    def __init__(self, models):
        self.n_column = len(models)
        self.columns = models
        self.y_ground_truth = models[models.keys()[0]][0].y.T[0]
        # self.y_test = np.zeros(shape=self.y_ground_truth.shape)

    def get_prediction(self):
        batch_size = 128
        dataset_size = 10000
        for column in self.columns.values():
            i = 0
            while i < dataset_size:
                batch_start = i
                batch_end = i+batch_size-1 if i+batch_size-1 < dataset_size-1 else dataset_size-1

                x_batch, y_batch = get_nparray_from_design_matrix(column[0], batch_start, batch_end)
                # x_batch, y_batch = get_nparray_from_design_matrix(column[0], 0, 127)

                f_model = column[1]
                y = f_model(x_batch)

                column[2][batch_start:batch_end] = y# np.argmax(y, axis=1)
                print batch_start, ':', batch_end, '   ', get_statistics(y_batch, y)
                i += batch_size


        # print self.y_ground_truth.shape, self.y_test.shape
            print get_statistics(self.y_ground_truth, column[2])
#
#
# def average_dnn_results(dnn_predictors, x_test, y_test):
#     y_avg = np.zeros(y_test.shape[1], 10)
#
#     for i, predictor in enumerate(predictor_list):
#         single_dnn = get_prediction(predictor, x_column0, y_column0)
#         print 'column ', i, 'results: '
#         print '/t', get_statistics(single_dnn)
#         y_avg += single_dnn
#
#     y_avg /= len(dnn_predictors)
#     pass




    # return predictor_list


if __name__ == '__main__':
    # get dataset CIFAR10

    cifar10_gcn = CIFAR10(which_set='test',
                             gcn=1,
                             axes=['c', 0, 1, 'b'])
    cifar10_toronto = CIFAR10(which_set='test',
                             toronto_prepro=True,
                             axes=['c', 0, 1, 'b'])

    columns = {
        # 'gcn': (cifar10_gcn, load_model_from_pkl('pkl/toronto_best.pkl')),
        'toronto': (cifar10_toronto, load_model_from_pkl('pkl/toronto_best.pkl'), np.zeros((10000, 10)))
    }

    multi_column_dnn = MCDNN(columns)


    multi_column_dnn.get_prediction()

    # predictor_list = get_predictor(models_path)



