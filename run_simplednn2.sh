#!/bin/bash

THEANO_FLAGS="device=gpu,floatX=float32" python train_simplednn2.py 2>&1 | tee ./output/simple_dnn2.txt
