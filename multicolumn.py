from pylearn2.datasets.cifar10 import CIFAR10
from pylearn2.space import Conv2DSpace
from pylearn2.utils import serial
import theano
from theano import tensor as T
import numpy as np



class MCDNN():

    def __init__(self, models):
        self.n_column = 2
        self.columns = {}
        self.generate_test_sets(models)


        models_path = ['pkl/toronto_best.pkl']

    def get_predictor(self, model_path):

        # get model
        model = serial.load(model_path)
        X = model.get_input_space().make_theano_batch()
        Y = model.fprop(X)
        # get prediction
        return theano.function([X], Y)


    def generate_test_sets(self, columns):

        for key in columns.keys():
            if key == 'gcn':
                cifar10_gcn = CIFAR10(which_set='test',
                             gcn=1,
                             axes=['c', 0, 1, 'b'])

                self.columns['gcn'] = (cifar10_gcn, self.get_predictor(columns[key]))
            if key == 'toronto':
                cifar10_toronto = CIFAR10(which_set='test',
                             toronto_prepro=True,
                             axes=['c', 0, 1, 'b'])
                self.columns['toronto'] = (cifar10_toronto,  self.get_predictor(columns[key]))

        # x_column0, y_column0 = get_nparray_from_design_matrix(cifar10_gcn, start, stop)
        # x_column1, y_column1 = get_nparray_from_design_matrix(cifar10_toronto, start, stop)
        # self.y = y_column0





def get_nparray_from_design_matrix(dataset, start=0, stop=128):
    """
    get design matrix in form batch per image
    images are in form [c, 0, 1]
    """
    x_dataset = dataset.X[start:stop, :]
    bc01_shape = (stop-start, 3, 32, 32)
    x_dataset = x_dataset.reshape(bc01_shape)
    axis_order = [('b', 'c', 0, 1).index(axis) for axis in ['c', 0, 1, 'b']]

    y_dataset = dataset.y[start:stop, :]
    # print y_dataset, y_dataset.shape
    return x_dataset.transpose(*axis_order), y_dataset.T[0]


def get_prediction(predictor, x_test, y_test):
    return predictor(x_test)


def get_statistics(y_true, prediction):
    y_hat = np.argmax(prediction, axis=1)
    misclass_mean = np.not_equal(y_hat, y_true).mean()
    misclass_var = np.not_equal(y_hat, y_true).var()
    return misclass_mean, misclass_var


def average_dnn_results(dnn_predictors, x_test, y_test):
    y_avg = np.zeros(y_test.shape[1], 10)

    for i, predictor in enumerate(predictor_list):
        single_dnn = get_prediction(predictor, x_column0, y_column0)
        print 'column ', i, 'results: '
        print '/t', get_statistics(single_dnn)
        y_avg += single_dnn

    y_avg /= len(dnn_predictors)
    pass




    return predictor_list


if __name__ == '__main__':
    # get dataset CIFAR10

    columns = {'gcn': 'pkl/toronto_best.pkl'}


    start = 0
    stop = 100


    print y_column0==y_column1


    # predictor_list = get_predictor(models_path)






