#!/bin/bash

THEANO_FLAGS="device=gpu,floatX=float32" python train_multicolumn_3COL_complex.py 2>&1 | tee output/multicolumn_3COL_complex.txt
