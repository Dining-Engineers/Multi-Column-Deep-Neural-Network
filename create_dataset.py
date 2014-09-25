"""
This script makes a dataset of 32x32 approximately whitened CIFAR-10 images.

"""
import cPickle
from cPickle import Pickler
from pylearn2.datasets.vector_spaces_dataset import VectorSpacesDataset
from pylearn2.datasets.zca_dataset import ZCA_Dataset
from pylearn2.format.target_format import OneHotFormatter
from pylearn2.space import CompositeSpace, Conv2DSpace, VectorSpace
import theano

from pylearn2.utils import serial
from pylearn2.datasets import preprocessing
from pylearn2.utils import string_utils
import numpy as np
from pylearn2.datasets.cifar10 import CIFAR10
import pylearn2.datasets.vector_spaces_dataset

data_dir = string_utils.preprocess('${PYLEARN2_DATA_PATH}/cifar10')


# input_space = VectorSpace(dim=3072, dtype=theano.config.floatX)
# target_space = VectorSpace(dim=10)

train_set_dimension = 5
n_classes = 10

print 'Loading CIFAR-10 train dataset...'
train_orig = CIFAR10(which_set='train',
                     start=0,
                     stop=train_set_dimension,
                     # axes=['b', 0, 1, 'c'])
                     axes=['c', 0, 1, 'b'])

# with open(data_dir+"/pylearn2_gcn_whitened/train.pkl", 'rb') as f:
#     pre_dataset = cPickle.load(f)
#
# with open(data_dir+"/pylearn2_gcn_whitened/preprocessor.pkl", 'rb') as f:
#     preproc = cPickle.load(f)
#
# train_white = ZCA_Dataset(preprocessed_dataset=pre_dataset,
#                           preprocessor=preproc,
#                           start=0,
#                           stop=50000,
#                           axes=['c', 0, 1, 'b'])

print "sblinda"

# print train_orig.get_data_specs()
# train_orig.data_specs = (CompositeSpace((input_space, target_space)), ("features", "targets"))
# print train_orig._iter_data_specs
# print train_orig.get_data()



print "Preparing output directory..."
output_dir = data_dir + '/pylearn2_whitened'
serial.mkdir( output_dir )

# print train_orig.y, train_orig.y_labels
# print train_orig.get_topological_view(train_orig.y)


t_orig_x = train_orig.get_topological_view()# train_orig.X.reshape(3, 32, 32, train_set_dimension)
print t_orig_x.shape
t_orig_y = OneHotFormatter(n_classes).format(train_orig.y, mode="concatenate")


#
# for elem in list(t_orig_y):
#     print elem.shape
#
# print len(set(elem.shape[0] for elem in list(t_orig_y)))
#
# print t_orig_x.shape
# print t_orig_y.shape

# print np.random.rand(3, 32, 32, train_set_dimension)


# c = VectorSpacesDataset(
#         (np.random.rand(20, 20).astype(theano.config.floatX),
#          np.random.rand(20, 15).astype(theano.config.floatX),
#          np.random.rand(20, 5).astype(theano.config.floatX)),
#         (CompositeSpace([
#             VectorSpace(20),
#             VectorSpace(15),
#             VectorSpace(5)]),
#         ('features1', 'features0', 'targets'))
#     )

b = VectorSpacesDataset(
        (np.random.rand(3, 32, 32, train_set_dimension).astype(theano.config.floatX),
         np.random.rand(3, 32, 32, train_set_dimension).astype(theano.config.floatX),
         np.random.rand(train_set_dimension, 5).astype(theano.config.floatX)),
        (CompositeSpace([
            Conv2DSpace(shape=(32, 32), num_channels=3, axes=('c', 0, 1, 'b')),
            Conv2DSpace(shape=(32, 32), num_channels=3, axes=('c', 0, 1, 'b')),
            VectorSpace(5)]),
        ('features1', 'features0', 'targets'))
)


#
# train = VectorSpacesDataset(
#         (t_orig_x.astype(theano.config.floatX),
#          t_orig_x.astype(theano.config.floatX),
#          t_orig_y.astype(theano.config.floatX)),
#         (CompositeSpace([
#             Conv2DSpace(shape=(32, 32), num_channels=3, axes=('c', 0, 1, 'b')),
#             Conv2DSpace(shape=(32, 32), num_channels=3, axes=('c', 0, 1, 'b')),
#             VectorSpace(n_classes)]),
#         ('features1', 'features0', 'targets')))
#



#
#
#
# print "Learning the preprocessor and preprocessing the unsupervised train data..."
# preprocessor = preprocessing.ZCA()
# train.apply_preprocessor(preprocessor = preprocessor, can_fit = True)
#
#
#
# print 'Saving the unsupervised data'
# train.use_design_loc(output_dir+'/train.npy')
# serial.save(output_dir + '/train.pkl', train)
#
# print "Loading the test data"
# test = CIFAR10(which_set = 'test')
#
# print "Preprocessing the test data"
# test.apply_preprocessor(preprocessor = preprocessor, can_fit = False)
#
# print "Saving the test data"
# test.use_design_loc(output_dir+'/test.npy')
# serial.save(output_dir+'/test.pkl', test)
#
# serial.save(output_dir + '/preprocessor.pkl',preprocessor)
