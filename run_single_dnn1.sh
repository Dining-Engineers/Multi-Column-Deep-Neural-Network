#!/bin/bash

THEANO_FLAGS="device=gpu,floatX=float32" python train_sdnn.py 2>&1 | tee ./output/single_dnn1.txt
