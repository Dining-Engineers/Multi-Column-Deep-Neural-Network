#!/bin/bash

THEANO_FLAGS="device=gpu,floatX=float32" python train_multicolumn_complex.py 2>&1 | tee output/multicolumn_complex.txt
