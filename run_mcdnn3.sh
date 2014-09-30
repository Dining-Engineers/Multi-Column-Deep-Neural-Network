#!/bin/bash

THEANO_FLAGS="device=gpu,floatX=float32" python train_MCDNN3.py 2>&1 | tee ./output/mc_dnn3.txt
