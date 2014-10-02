from itertools import product

import numpy as np
from pylearn2.costs.mlp.dropout import Dropout
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
from pylearn2.utils import is_iterable, sharedX, serial
from pylearn2.expr.nnet import pseudoinverse_softmax_numpy
from custom_layers import Average


"""
Create a VectorSpacesDataset with two inputs (features0 and features1)
and train an MLP which takes both inputs for 1 epoch.
"""
mlp = MLP(
    layers=[
        CompositeLayer(
            layer_name='composite',
            layers=
            [
                MLP(layer_name='mlp1', layers=[Linear(10, 'h0', 0.1)]),
                Linear(10, 'h1', 0.1)
            ],
            inputs_to_layers=
            {
                0: [0],
                1: [1]
            }
        ),
        Average('sum'),
        Softmax(10, 'softmax', 0.1)
    ],
    input_space=CompositeSpace([VectorSpace(15), VectorSpace(20)]),
    input_source=('features0', 'features1')
)

dataset = VectorSpacesDataset(
    (np.random.rand(20, 20).astype(theano.config.floatX),
     np.random.rand(20, 15).astype(theano.config.floatX),
     np.random.rand(20, 5).astype(theano.config.floatX)),
    (CompositeSpace([
        VectorSpace(20),
        VectorSpace(15),
        VectorSpace(5)]),
     ('features1', 'features0', 'targets'))
)
train = Train(dataset, mlp, SGD(0.1, batch_size=5, monitoring_dataset=dataset,
                                cost=Dropout(
                                    input_include_probs={'composite_mlp1': 1, },
                                    input_scales={'composite_mlp1': 1, }
                                ),
))

# # Load the saved model
# model = serial.load(saved_model_path)
#
# # Remove last layer
# del model.layers[-1]
#
# # Add new layer
# new_output_layer = <make your new layer here>
# model.add_layers([new_output_layer])


column1_pretrained = serial.load('./pkl/multicolumn_complex.pkl')
pretrained_layers_1 = column1_pretrained.layers

print pretrained_layers_1, len(pretrained_layers_1)


# mlp.layers.extend(pretrained_layers[start_layer:])

# , cost=Dropout(input_include_probs={'composite':1.})))
# train.algorithm.termination_criterion = EpochCounter(1)
# train.main_loop()