#!/bin/bash

THEANO_FLAGS="device=gpu,floatX=float32" python train_sdnn2.py 2>&1 | tee ./output/single_dnn2.txt
