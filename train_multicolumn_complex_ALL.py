from pylearn2.config import yaml_parse

print "#####################################################"
print " TRAIN 2 COL GCN TOR"

with open("yaml/multicolumn_DNN_2COL_GCN_TOR.yaml", 'r') as f:
    train = f.read()
train = yaml_parse.load(train)
train.main_loop()

print "#####################################################"
print " TRAIN 2 COL ZCA TOR"

with open("yaml/multicolumn_DNN_2COL_ZCA_TOR.yaml", 'r') as f:
    train = f.read()
train = yaml_parse.load(train)
train.main_loop()

print "#####################################################"
print " TRAIN 2 COL GCN ZCA"

with open("yaml/multicolumn_DNN_2COL_GCN_ZCA.yaml", 'r') as f:
    train = f.read()
train = yaml_parse.load(train)
train.main_loop()

print "#####################################################"
print " TRAIN 3 COL ALL"

with open("yaml/multicolumn_DNN_3COL.yaml", 'r') as f:
    train = f.read()
train = yaml_parse.load(train)
train.main_loop()