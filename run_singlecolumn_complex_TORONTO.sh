#!/bin/bash

THEANO_FLAGS="device=gpu,floatX=float32" python train_singlecolumn_complex_TORONTO.py 2>&1 | tee ./output/singlecolumn_complex_TORONTO.txt
