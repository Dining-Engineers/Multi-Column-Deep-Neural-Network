from pylearn2.config import yaml_parse

with open("yaml/multicolumn_DNN_2COL_GCN_TOR_NOFIXED.yaml", 'r') as f:
    train = f.read()
train = yaml_parse.load(train)
train.main_loop()