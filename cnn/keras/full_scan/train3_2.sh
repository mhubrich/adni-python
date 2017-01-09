#!/bin/bash

THEANO_FLAGS=device=gpu,floatX=float32 python train3.py 6 > output_3_CV6
THEANO_FLAGS=device=gpu,floatX=float32 python train3.py 7 > output_3_CV7
THEANO_FLAGS=device=gpu,floatX=float32 python train3.py 8 > output_3_CV8
THEANO_FLAGS=device=gpu,floatX=float32 python train3.py 9 > output_3_CV9
THEANO_FLAGS=device=gpu,floatX=float32 python train3.py 10 > output_3_CV10
