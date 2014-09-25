import os
import pylearn2
# path = os.path.join(pylearn2.__path__[0], 'scripts', 'tutorials', 'multilayer_perceptron', 'mlp_tutorial_part_4.yaml')
# with open(path, 'r') as f:
#     train_3 = f.read()
# hyper_params = {'train_stop' : 50000,
#                 'valid_stop' : 60000,
#                 'dim_h0' : 500,
#                 'dim_h1' : 1000,
#                 'sparse_init_h1' : 15,
#                 'max_epochs' : 10000,
#                 'save_path' : '.'}
# train_3 = train_3 % (hyper_params)
# print train_3



from pylearn2.config import yaml_parse

with open("./single_dnn.yaml", 'r') as f:
    train = f.read()
train = yaml_parse.load(train)
train.main_loop()