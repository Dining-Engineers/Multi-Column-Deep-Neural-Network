"""
This script makes a dataset of 32x32 approximately whitened CIFAR-10 images.

"""
from pylearn2.costs.mlp.dropout import Dropout
from pylearn2.datasets.vector_spaces_dataset import VectorSpacesDataset
from pylearn2.datasets.zca_dataset import ZCA_Dataset
from pylearn2.format.target_format import OneHotFormatter
from pylearn2.models.maxout import Maxout, MaxoutConvC01B
from pylearn2.space import CompositeSpace, Conv2DSpace, VectorSpace
from pylearn2.utils import string_utils
from pylearn2.datasets.cifar10 import CIFAR10
from vector_spaces_dataset_c01b import VectorSpacesDatasetC01B
from pylearn2.utils import is_iterable, sharedX, serial

def load_dataset(which_set, size='big'):

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
            stop_set = 10000
        elif which_set == 'valid':
            which_set = 'train'
            start_set = 40000
            stop_set = 43000
        else:
            #train
            start_set = 0
            stop_set = 10000

    n_classes = 10
    # take original cifar10dataset
    train_orig = CIFAR10(which_set=which_set,
                         start=start_set,
                         stop=stop_set,
                         gcn=55.,
                         axes=['b', 0, 1, 'c'])
                         # axes=['c', 0, 1, 'b'])

    # train_2 = CIFAR10(which_set=which_set,
    #                    start=start_set,
    #                    stop=stop_set,
    #                    axes=['b', 0, 1, 'c'],
    #                    # axes=['c', 0, 1, 'b'],
    #                    toronto_prepro=1,
    #                     )

    data_dir = string_utils.preprocess('${PYLEARN2_DATA_PATH}/cifar10')
    # IF ZCA
    train_2 = ZCA_Dataset(preprocessed_dataset=serial.load(data_dir+"/pylearn2_gcn_whitened/"+which_set+".pkl"),
                              preprocessor=serial.load(data_dir+"/pylearn2_gcn_whitened/preprocessor.pkl"),
                              start=start_set,
                              stop=stop_set,
                              axes=['b', 0, 1, 'c'])
                              # axes=['c', 0, 1, 'b'])
    #


    # ERROR: A design matrix with data in rows. Data is assumed to be laid out in
    #      memory according to the axis order ('b', 'c', 0, 1)
    # MORTEMORTEMORTE

    # # b01c input space
    input_space1 = Conv2DSpace(shape=(32, 32), num_channels=3, axes=('b', 0, 1, 'c'))
    input_space2 = Conv2DSpace(shape=(32, 32), num_channels=3, axes=('b', 0, 1, 'c'))
    # # c01b input space
    # input_space1 = Conv2DSpace(shape=(32, 32), num_channels=3, axes=('c', 0, 1, 'b'))
    # input_space2 = Conv2DSpace(shape=(32, 32), num_channels=3, axes=('c', 0, 1, 'b'))
    #
    # # Output Space
    out_space = VectorSpace(n_classes)


    # input1 = train_orig.get_formatted_view(train_orig, input_space1)
    input1 = train_orig.get_topological_view()  # train_orig.X.reshape(3, 32, 32, train_set_dimension)
    # input2 = train_2.get_formatted_view(train_orig, input_space2)
    input2 = train_2.get_topological_view()
    # print input1.shape
    target = OneHotFormatter(n_classes).format(train_orig.y, mode="concatenate")
    # print input1.shape, input2.shape, target.shape

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

    set = VectorSpacesDataset(
        (input1,
         input2,
         target),
        (CompositeSpace([
            input_space1,
            input_space2,
            out_space]),
         ('features0', 'features1', 'targets'))
    )

    # set = VectorSpacesDatasetC01B(
    #     (input1,
    #      input2,
    #      target),
    #     (CompositeSpace([
    #         input_space1,
    #         input_space2,
    #         out_space]),
    #      ('features0', 'features1', 'targets'))
    # )

    # print set.get_data()[1].shape
    return set


if __name__ == '__main__':
    load_dataset('train')



































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
