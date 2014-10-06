from pylearn2.train_extensions.best_params import MonitorBasedSaveBest
from pylearn2.termination_criteria import EpochCounter
from pylearn2.training_algorithms.sgd import SGD
from pylearn2.train import Train
from pylearn2.models.mlp import (FlattenerLayer, MLP, Softmax, CompositeLayer)
from pylearn2.space import CompositeSpace, Conv2DSpace
from pylearn2.utils import serial

import create_dataset
from custom_layers import PretrainedMLP, SpaceConverter2, Average


"""
Create a VectorSpacesDataset with two inputs (features0 and features1)
and train an MLP which takes both inputs
"""
mlp = MLP(
    batch_size=128,
    layers=[
        # FlattenerLayer(
        CompositeLayer(
            layer_name='composite',
            layers=
            [
                MLP(layers=[
                    SpaceConverter2('spconveter',
                                    Conv2DSpace(shape=[32, 32],
                                                num_channels=3,
                                                axes=['c', 0, 1, 'b']
                                    ))
                    ,
                    PretrainedMLP(layer_name='gcn_column',
                                     layer_content=serial.load('pkl/best/singlecolumn_complex_GCN_paper_best.pkl')),
                ]),

                MLP(layers=[
                    SpaceConverter2('spconveter',
                                    Conv2DSpace(shape=[32, 32],
                                                num_channels=3,
                                                axes=['c', 0, 1, 'b']
                                    ))
                    ,
                    # PretrainedColumn(layer_name='toronto_column',
                    #                  layer_content=serial.load('pkl/best/singlecolumn_complex_TORONTO_paper_best.pkl')),
                    PretrainedMLP(layer_name='zca_column',
                                     layer_content=serial.load('pkl/best/singlecolumn_complex_ZCA_paper_best.pkl')),
                ]),

            ],
            inputs_to_layers=
            {
                0: [0],
                1: [1]
            }
        # ),
        ),
        Average('avg'),
        Softmax(10, 'y', 0.05)
    ],
    input_space=CompositeSpace([
        Conv2DSpace(shape=(32, 32), num_channels=3, axes=('b', 0, 1, 'c')),
        Conv2DSpace(shape=(32, 32), num_channels=3, axes=('b', 0, 1, 'c'))]),
    input_source=('featureGCN', 'featureZCA')
)

dataset = create_dataset.load_dataset(which_set='train', dataset_types=['gcn', 'zca'])
dataset_valid = create_dataset.load_dataset(which_set='valid', dataset_types=['gcn', 'zca'])
dataset_test = create_dataset.load_dataset(which_set='test', dataset_types=['gcn', 'zca'])

train = Train(
    dataset,
    mlp,
    SGD(
        0.1,
        batch_size=128,
        monitoring_dataset={'train': dataset, 'valid': dataset_valid, 'test': dataset_test},
        termination_criterion=EpochCounter(100),
        train_iteration_mode='even_shuffled_sequential',
        monitor_iteration_mode='even_sequential'
    ),
    save_path="pkl/multicolumn.pkl",
    save_freq=5,
    extensions=[
        MonitorBasedSaveBest(
            channel_name='test_y_misclass',
            save_path="pkl/multicolumn_best.pkl"
        )
    ]

)

# # Load the saved model
# model = serial.load(saved_model_path)
#
# # Remove last layer
# del model.layers[-1]
#
# # Add new layer
# new_output_layer = <make your new layer here>
# model.add_layers([new_output_layer])


# mlp.layers.extend(pretrained_layers[start_layer:])

# , cost=Dropout(input_include_probs={'composite':1.})))
# train.algorithm.termination_criterion = EpochCounter(1)
train.main_loop()