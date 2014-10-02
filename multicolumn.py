from pylearn2.datasets.cifar10 import CIFAR10
from pylearn2.utils import serial
import theano
from theano import tensor as T


#
# # get dataset
# dataset = CIFAR10(which_set='test',
#                          start=0,
#                          stop=10000,
#                          # gcn=1,
#                          toronto_prepro=True,
#                          axes=['c', 0, 1, 'b'])
#
# x_test = dataset.get_topological_view()
#
# # print x_test[0,:]
#
# # number of column
# n_column = 2




# get model1
model_path = 'pkl/toronto_best.pkl'
model = serial.load( model_path )

X = model.get_input_space().make_theano_batch(batch_size=128)
Y = model.fprop( X )

# get prediction
f = theano.function( [X], Y )

y = f( x_test )

print y









# get results
Y = T.argmax( Y, axis = 1 )
