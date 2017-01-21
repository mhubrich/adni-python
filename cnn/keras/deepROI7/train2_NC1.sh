#!/bin/bash

THEANO_FLAGS=device=gpu,floatX=float32,lib.cnmem=1.0 python train_AE_NC_2.py 1 > output_AE_NC_2_CV1
THEANO_FLAGS=device=gpu,floatX=float32,lib.cnmem=1.0 python train_AE_NC_2.py 2 > output_AE_NC_2_CV2
THEANO_FLAGS=device=gpu,floatX=float32,lib.cnmem=1.0 python train_AE_NC_2.py 3 > output_AE_NC_2_CV3

