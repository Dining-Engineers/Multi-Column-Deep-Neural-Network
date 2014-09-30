#!/bin/bash

THEANO_FLAGS="device=gpu,floatX=float32" python train_sdnn3.py 2>&1 | tee ./output/single_dnn3.txt
