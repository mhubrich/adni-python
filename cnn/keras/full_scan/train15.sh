#!/bin/bash

THEANO_FLAGS=device=gpu,floatX=float32 python train15.py 1 > output_15_CV1
THEANO_FLAGS=device=gpu,floatX=float32 python train15.py 2 > output_15_CV2
THEANO_FLAGS=device=gpu,floatX=float32 python train15.py 3 > output_15_CV3
THEANO_FLAGS=device=gpu,floatX=float32 python train15.py 4 > output_15_CV4
THEANO_FLAGS=device=gpu,floatX=float32 python train15.py 5 > output_15_CV5
THEANO_FLAGS=device=gpu,floatX=float32 python train15.py 6 > output_15_CV6
THEANO_FLAGS=device=gpu,floatX=float32 python train15.py 7 > output_15_CV7
THEANO_FLAGS=device=gpu,floatX=float32 python train15.py 8 > output_15_CV8
THEANO_FLAGS=device=gpu,floatX=float32 python train15.py 9 > output_15_CV9
THEANO_FLAGS=device=gpu,floatX=float32 python train15.py 10 > output_15_CV10
