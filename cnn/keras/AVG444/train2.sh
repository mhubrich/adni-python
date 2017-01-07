#!/bin/bash

THEANO_FLAGS=device=gpu,floatX=float32 python train.py 6 > output_AVG444_3_CV6
THEANO_FLAGS=device=gpu,floatX=float32 python train.py 7 > output_AVG444_3_CV7
THEANO_FLAGS=device=gpu,floatX=float32 python train.py 8 > output_AVG444_3_CV8
THEANO_FLAGS=device=gpu,floatX=float32 python train.py 9 > output_AVG444_3_CV9
THEANO_FLAGS=device=gpu,floatX=float32 python train.py 10 > output_AVG444_3_CV10

