#!/bin/bash


THEANO_FLAGS="device=gpu,floatX=float32" python train_multicolumn_complex_2COL_GCN_TOR_NOFIXED.py 2>&1 | tee output/train_multicolumn_complex_2COL_GCN_TOR_NOFIXED.txt
#THEANO_FLAGS="device=gpu,floatX=float32" python train_multicolumn_complex_2COL_GCN_ZCA_NOFIXED.py 2>&1 | tee output/train_multicolumn_complex_2COL_GCN_ZCA_NOFIXED.txt
#THEANO_FLAGS="device=gpu,floatX=float32" python train_multicolumn_complex_2COL_ZCA_TOR_NOFIXED.py 2>&1 | tee output/train_multicolumn_complex_2COL_ZCA_TOR_NOFIXED.txt
#THEANO_FLAGS="device=gpu,floatX=float32" python train_multicolumn_3COL_complex_NOFIXED.py 2>&1 | tee output/train_multicolumn_3COL_complex_NOFIXED.txt
