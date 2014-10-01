#!/bin/bash

THEANO_FLAGS="device=gpu,floatX=float32" python train_singlecolumn_simple_TORONTO.py 2>&1 | tee ./output/singlecolumn_simple_TORONTO.txt
