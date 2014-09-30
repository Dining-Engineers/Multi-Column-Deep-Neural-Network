from itertools import product

import numpy as np
from pylearn2.costs.mlp.dropout import Dropout
from pylearn2.datasets.cifar10 import CIFAR10
from pylearn2.models.maxout import MaxoutConvC01B
from pylearn2.sandbox.cuda_convnet.debug import batch_size
import theano
from theano import tensor, config
from nose.tools import assert_raises

from pylearn2.datasets.vector_spaces_dataset import VectorSpacesDataset
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from pylearn2.termination_criteria import EpochCounter
from pylearn2.training_algorithms.sgd import SGD
from pylearn2.train import Train
from pylearn2.models.mlp import (FlattenerLayer, MLP, Linear, Softmax, Sigmoid,
                                 exhaustive_dropout_average,
                                 sampled_dropout_average, CompositeLayer)
from pylearn2.space import VectorSpace, CompositeSpace, Conv2DSpace
from pylearn2.utils import is_iterable, sharedX
from pylearn2.expr.nnet import pseudoinverse_softmax_numpy
from AverageLayer import Average


"""
Create a VectorSpacesDataset with two inputs (features0 and features1)
and train an MLP which takes both inputs for 1 epoch.
"""
mlp = MLP(
    layers=[
            MaxoutConvC01B(
                layer_name='conv1',
                pad= 0,
                num_channels= 32,
                num_pieces= 1,
                kernel_shape= [5, 5],
                pool_shape= [3, 3],
                pool_stride= [2, 2],
                irange= .01,
                min_zero= True,
                W_lr_scale= 1.,
                b_lr_scale= 2.,
                tied_b=True,
                max_kernel_norm=9.9,
            ),
        # Average('sum'),
        Softmax(10, 'softmax', 0.1)
    ],
    input_space=Conv2DSpace (
            shape= [32, 32],
            num_channels= 3,
            axes= ['c', 0, 1, 'b']),
)

dataset = CIFAR10(which_set='train',
                         start=0,
                         stop=40000,
                         # axes=['b', 0, 1, 'c'])
                         axes=['c', 0, 1, 'b'])


train = Train(dataset,
              mlp,
              SGD(
                  0.1, batch_size=5,
                    batch_size=128,
                    learning_rate= .01,
                    init_momentum= .9,
                    train_iteration_mode= 'even_shuffled_sequential',
                    monitor_iteration_mode= 'even_sequential',
                    monitoring_dataset=
                    {
                        'valid' : CIFAR10 (
                                      toronto_prepro= True,
                                      axes= ['c', 0, 1, 'b'],
                                      which_set= 'train',
                                      one_hot= 1,
                                      start= 40000,
                                      stop=  50000
                        ),
                        'test': CIFAR10 (
                                      toronto_prepro= True,
                                      axes= ['c', 0, 1, 'b'],
                                      which_set= 'test',
                                      one_hot= 1,
                                      ),
                    },
                    cost= Dropout (
                        input_include_probs= { 'conv1':.8 },
                        input_scales= {'conv1':1.}
                    ),
                    termination_criterion=EpochCounter(max_epochs= 5),

              )
)#, cost=Dropout(input_include_probs={'composite'=1.})))
train.algorithm.termination_criterion = EpochCounter(3)
train.main_loop()