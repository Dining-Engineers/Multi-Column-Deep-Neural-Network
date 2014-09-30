from pylearn2.costs.mlp.dropout import Dropout
from pylearn2.datasets.cifar10 import CIFAR10
from pylearn2.devtools.nan_guard import NanGuardMode
from pylearn2.models.maxout import MaxoutConvC01B

from pylearn2.termination_criteria import EpochCounter
from pylearn2.training_algorithms.sgd import SGD
from pylearn2.train import Train
from pylearn2.models.mlp import (MLP, Softmax, SpaceConverter, ConvRectifiedLinear)
from pylearn2.space import Conv2DSpace
import theano
from custom_layers import PreprocessorBlock


"""
Create a VectorSpacesDataset with two inputs (features0 and features1)
and train an MLP which takes both inputs for 1 epoch.
"""
mlp = MLP(
    batch_size=128,
    layers=[
        PreprocessorBlock(
            layer_name='preproc',
            # toronto=True,
            gcn=1
        ),
        SpaceConverter('spconveter',
                       Conv2DSpace(shape=[32, 32],
                                   num_channels=3,
                                   axes=['c', 0, 1, 'b']
                                   # axes= ['c', 0, 1, 'b']
                       )
        ),
        MaxoutConvC01B(
            layer_name='conv1',
            pad=0,
            num_channels=32,
            num_pieces=1,
            kernel_shape=[5, 5],
            pool_shape=[3, 3],
            pool_stride=[2, 2],
            irange=.01,
            min_zero=True,
            W_lr_scale=1.,
            b_lr_scale=2.,
            tied_b=True,
            max_kernel_norm=9.9,
        ),
        # ConvRectifiedLinear(layer_name='conv1',
        # irange= .01,
        #                     output_channels = 32,
        #                      kernel_shape = [5,5],
        #                      pool_shape = [3, 3],
        #                      pool_stride = [2, 2]
        #
        #                                ),
        # Average('sum'),
        Softmax(10, 'softmax', 0.1)
    ],
    input_space=Conv2DSpace(
        shape=[32, 32],
        num_channels=3,
        axes=['b', 0, 1, 'c']
        # axes= ['c', 0, 1, 'b']
    ),
)

dataset = CIFAR10(
    # toronto_prepro= True,
    which_set='train',
    one_hot=1,
    # axes= ['c', 0, 1, 'b'],
    axes=['b', 0, 1, 'c'],
    start=0,
    stop=50000)

train = Train(dataset=dataset,
              model=mlp,
              algorithm=SGD(
                  # monitor_iteration_mode=NanGuardMode(
                  #     nan_is_error=True,
                  #     inf_is_error=True
                  # ),
                  batch_size=128,
                  learning_rate=.01,
                  init_momentum=.9,
                  train_iteration_mode='even_shuffled_sequential',
                  monitor_iteration_mode= 'even_sequential',
                  monitoring_dataset=
                  {
                      'valid': CIFAR10(
                          #toronto_prepro= True,
                          axes=['c', 0, 1, 'b'],
                          which_set='train',
                          one_hot=1,
                          start=40000,
                          stop=50000
                      ),
                      'test': CIFAR10(
                          #toronto_prepro= True,
                          axes=['c', 0, 1, 'b'],
                          which_set='test',
                          one_hot=1,
                      ),
                  },
                  cost=Dropout(
                      input_include_probs={'conv1': .8},
                      input_scales={'conv1': 1.}
                  ),
                  termination_criterion=EpochCounter(max_epochs=100),

              )
)  # , cost=Dropout(input_include_probs={'composite'=1.})))
train.main_loop()

#
# # I tried to follow your advice and write this code:
# X = train.model.get_input_space().make_theano_batch()
# Y =
# Y = train.model.fprop(X)
# f = theano.function([X], Y)
# f([[[[0.5], [0.5]]]])
# #array([[ 0.49999138,  0.50000862]])
#
# print(y)