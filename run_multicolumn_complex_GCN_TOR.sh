#!/bin/bash

THEANO_FLAGS="device=gpu,floatX=float32" python train_multicolumn_complex_2COL_GCN_TOR.py 2>&1 | tee output/train_multicolumn_complex_2COL_GCN_TOR.txt
