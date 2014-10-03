import theano
from pylearn2.utils import serial
import numpy as np


def load_model_from_pkl(model_path):

        # get model
        model = serial.load(model_path)
        X = model.get_input_space().make_theano_batch()
        Y = model.fprop(X)
        # get prediction
        return theano.function([X], Y)

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


def get_statistics(y_true, prediction):
    y_hat = np.argmax(prediction, axis=1)
    misclass_mean = np.not_equal(y_hat, y_true).mean()
    misclass_var = np.not_equal(y_hat, y_true).var()
    return misclass_mean, misclass_var