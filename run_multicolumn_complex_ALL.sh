#!/bin/bash


THEANO_FLAGS="device=gpu,floatX=float32" python train_multicolumn_complex_2COL_GCN_TOR.py 2>&1 | tee output/train_multicolumn_complex_2COL_GCN_TOR.txt
THEANO_FLAGS="device=gpu,floatX=float32" python train_multicolumn_complex_2COL_GCN_ZCA.py 2>&1 | tee output/train_multicolumn_complex_2COL_GCN_ZCA.txt
THEANO_FLAGS="device=gpu,floatX=float32" python train_multicolumn_complex_2COL_ZCA_TOR.py 2>&1 | tee output/train_multicolumn_complex_2COL_ZCA_TOR.txt
THEANO_FLAGS="device=gpu,floatX=float32" python train_multicolumn_3COL_complex.py 2>&1 | tee output/train_multicolumn_3COL_complex.txt
