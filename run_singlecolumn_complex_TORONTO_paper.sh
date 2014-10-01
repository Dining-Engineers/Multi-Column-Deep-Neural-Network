#!/bin/bash

THEANO_FLAGS="device=gpu,floatX=float32" python train_singlecolumn_complex_TORONTO_paper.py 2>&1 | tee ./output/singlecolumn_complex_TORONTO_paper.txt
