from pylearn2.datasets.cifar10 import CIFAR10
from pylearn2.space import Conv2DSpace
from pylearn2.utils import serial
import theano
from theano import tensor as T



# get dataset
dataset = CIFAR10(which_set='train',
                         start=0,
                         stop=2,
                         # gcn=1,
                         toronto_prepro=True,
                         axes=['c', 0, 1, 'b'])


num_test_examples = 2
# get design matrix in form batch per image
# images are in form [c, 0, 1]
x_dataset = dataset.X # bc01

x_dataset2 = x_dataset.copy()
bc01_shape = (num_test_examples, 3, 32, 32)
x_dataset2.reshape(bc01_shape)
print x_dataset2.shape
axis_order = [('b', 'c', 0, 1).index(axis) for axis in ['c', 0, 1, 'b']]
print axis_order

x_dataset3 = x_dataset2.transpose(1,2,3,0)




# print dataset.axes



dataset.set_view_converter_axes(['c', 0, 1, 'b'])
x_test = dataset.get_topological_view()


print x_dataset2==x_test

#
# print 'x', dataset.X, dataset.X.shape
# print 'x test', x_test
#
# # number of column
# n_column = 2
#
# # get model1
# model_path = 'pkl/toronto_best.pkl'
# model = serial.load(model_path)
#
#
# # input_space1 = Conv2DSpace(shape=(32, 32), num_channels=3, axes=('b', 0, 1, 'c'))
# # x_input = dataset.get_formatted_view(dataset.X, input_space1)
#
# X = model.get_input_space().make_theano_batch(batch_size=128)
# Y = model.fprop(X)
#
# # get prediction
# f = theano.function([X], Y)
#
# y = f(x_test)
#
# print 'prediction', y
#
# # get results
# Y = T.argmax(Y, axis=1)
#
# print 'result', Y.get_value()
# print 'true result', dataset.y[0]