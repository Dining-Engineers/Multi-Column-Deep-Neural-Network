"""
This script makes a dataset of 32x32 approximately whitened CIFAR-10 images.

"""
from pylearn2.costs.mlp.dropout import Dropout
from pylearn2.datasets.vector_spaces_dataset import VectorSpacesDataset
from pylearn2.format.target_format import OneHotFormatter
from pylearn2.models.maxout import Maxout, MaxoutConvC01B
from pylearn2.space import CompositeSpace, Conv2DSpace, VectorSpace
from pylearn2.utils import string_utils
from pylearn2.datasets.cifar10 import CIFAR10
from vector_spaces_dataset_c01b import VectorSpacesDatasetC01B


def load_dataset(which_set):

    size = 'small'
    print "loading.. ", which_set


    if size == 'big':
        if which_set == 'test':
            start_set = 0
            stop_set = 10000
        elif which_set == 'valid':
            which_set = 'train'
            start_set = 40000
            stop_set = 50000
        else:
            #train
            start_set = 0
            stop_set = 40000
    else:
        if which_set == 'test':
            start_set = 0
            stop_set = 4000
        elif which_set == 'valid':
            which_set = 'train'
            start_set = 40000
            stop_set = 43000
        else:
            #train
            start_set = 0
            stop_set = 10000

    data_dir = string_utils.preprocess('${PYLEARN2_DATA_PATH}/cifar10')

    n_classes = 10

    # take original cifar10dataset
    train_orig = CIFAR10(which_set=which_set,
                         start=start_set,
                         stop=stop_set,
                         # axes=['b', 0, 1, 'c'])
                         axes=['c', 0, 1, 'b'])

    train_2  = CIFAR10(which_set=which_set,
                       start=start_set,
                       stop=stop_set,
                       # axes=['b', 0, 1, 'c'],
                       axes=['c', 0, 1, 'b'],
                       toronto_prepro=1,
                       # gcn=1
                        )


    # print t2.X[1, :] == train_orig.X[1, :]
                         # axes=['c', 0, 1, 'b'])


    # IF ZCA
    # # take whitened cifar10 dataset
    # with open(data_dir+"/pylearn2_gcn_whitened/train.pkl", 'rb') as f:
    #     pre_dataset = cPickle.load(f)
    #
    # with open(data_dir+"/pylearn2_gcn_whitened/preprocessor.pkl", 'rb') as f:
    #     preproc = cPickle.load(f)
    #
    # train_2 = ZCA_Dataset(preprocessed_dataset=pre_dataset,
    #                           preprocessor=preproc,
    #                           start=start_set,
    #                           stop=stop_set,
    #                           axes=['b', 0, 1, 'c'])
    #                           # axes=['c', 0, 1, 'b'])
    #



    input1 = train_orig.get_topological_view()  # train_orig.X.reshape(3, 32, 32, train_set_dimension)
    input2 = train_2.get_topological_view()
    target = OneHotFormatter(n_classes).format(train_orig.y, mode="concatenate")


    # # b01c input space
    # input_space1 = Conv2DSpace(shape=(32, 32), num_channels=3, axes=('b', 0, 1, 'c'))
    # input_space2 = Conv2DSpace(shape=(32, 32), num_channels=3, axes=('b', 0, 1, 'c'))
    # # c01b input space
    input_space1 = Conv2DSpace(shape=(32, 32), num_channels=3, axes=('c', 0, 1, 'b'))
    input_space2 = Conv2DSpace(shape=(32, 32), num_channels=3, axes=('c', 0, 1, 'b'))
    #
    # # Output Space
    out_space = VectorSpace(n_classes)
    #
    #
    # ###### TEST ###############################################################################
    # # c01b
    # # input1 = np.random.rand(3, 32, 32, train_set_dimension).astype(theano.config.floatX)
    # # input2 = np.random.rand(3, 32, 32, train_set_dimension).astype(theano.config.floatX)
    # # target = np.random.rand(train_set_dimension, n_classes).astype(theano.config.floatX)
    # # input_space1 = Conv2DSpace(shape=(32, 32), num_channels=3, axes=('c', 0, 1, 'b'))
    # # input_space2 = Conv2DSpace(shape=(32, 32), num_channels=3, axes=('c', 0, 1, 'b'))
    #
    # #
    # # #b01c
    # # input1 = np.random.rand(train_set_dimension, 32, 32, 3).astype(theano.config.floatX)
    # # input2 = np.random.rand(train_set_dimension, 32, 32, 3).astype(theano.config.floatX)
    # # target = np.random.rand(train_set_dimension, n_classes).astype(theano.config.floatX)
    # # input_space1 = Conv2DSpace(shape=(32, 32), num_channels=3, axes=('b', 0, 1, 'c'))
    # # input_space2 = Conv2DSpace(shape=(32, 32), num_channels=3, axes=('b', 0, 1, 'c'))
    # ###########################################################################################
    #
    #

    set = VectorSpacesDatasetC01B(
        (input1,
         input2,
         target),
        (CompositeSpace([
            input_space1,
            input_space2,
            out_space]),
         ('features0', 'features1', 'targets'))
    )

    return set




































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
