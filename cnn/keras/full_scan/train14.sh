#!/bin/bash

THEANO_FLAGS=device=gpu,floatX=float32 python train14.py 1 > output_14_CV1
THEANO_FLAGS=device=gpu,floatX=float32 python train14.py 2 > output_14_CV2
THEANO_FLAGS=device=gpu,floatX=float32 python train14.py 3 > output_14_CV3
THEANO_FLAGS=device=gpu,floatX=float32 python train14.py 4 > output_14_CV4
THEANO_FLAGS=device=gpu,floatX=float32 python train14.py 5 > output_14_CV5
THEANO_FLAGS=device=gpu,floatX=float32 python train14.py 6 > output_14_CV6
THEANO_FLAGS=device=gpu,floatX=float32 python train14.py 7 > output_14_CV7
THEANO_FLAGS=device=gpu,floatX=float32 python train14.py 8 > output_14_CV8
THEANO_FLAGS=device=gpu,floatX=float32 python train14.py 9 > output_14_CV9
THEANO_FLAGS=device=gpu,floatX=float32 python train14.py 10 > output_14_CV10
