#!/bin/bash

THEANO_FLAGS=device=gpu,floatX=float32 python train.py 1 > output_first3_CV1
THEANO_FLAGS=device=gpu,floatX=float32 python train.py 2 > output_first3_CV2
THEANO_FLAGS=device=gpu,floatX=float32 python train.py 3 > output_first3_CV3
THEANO_FLAGS=device=gpu,floatX=float32 python train.py 4 > output_first3_CV4
THEANO_FLAGS=device=gpu,floatX=float32 python train.py 5 > output_first3_CV5
THEANO_FLAGS=device=gpu,floatX=float32 python train.py 6 > output_first3_CV6
THEANO_FLAGS=device=gpu,floatX=float32 python train.py 7 > output_first3_CV7
THEANO_FLAGS=device=gpu,floatX=float32 python train.py 8 > output_first3_CV8
THEANO_FLAGS=device=gpu,floatX=float32 python train.py 9 > output_first3_CV9
THEANO_FLAGS=device=gpu,floatX=float32 python train.py 10 > output_first3_CV10
