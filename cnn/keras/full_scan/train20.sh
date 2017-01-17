#!/bin/bash

#THEANO_FLAGS=device=gpu,floatX=float32 python train20.py 1 > output_20_CV1
#THEANO_FLAGS=device=gpu,floatX=float32 python train20.py 2 > output_20_CV2
#THEANO_FLAGS=device=gpu,floatX=float32 python train20.py 3 > output_20_CV3
THEANO_FLAGS=device=gpu,floatX=float32 python train20.py 4 > output_20_CV4
THEANO_FLAGS=device=gpu,floatX=float32 python train20.py 5 > output_20_CV5
THEANO_FLAGS=device=gpu,floatX=float32 python train20.py 6 > output_20_CV6
THEANO_FLAGS=device=gpu,floatX=float32 python train20.py 7 > output_20_CV7
THEANO_FLAGS=device=gpu,floatX=float32 python train20.py 8 > output_20_CV8
THEANO_FLAGS=device=gpu,floatX=float32 python train20.py 9 > output_20_CV9
THEANO_FLAGS=device=gpu,floatX=float32 python train20.py 10 > output_20_CV10
