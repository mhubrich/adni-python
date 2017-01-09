#!/bin/bash

THEANO_FLAGS=device=gpu,floatX=float32 python train6.py 1 > output_6_CV1
THEANO_FLAGS=device=gpu,floatX=float32 python train6.py 2 > output_6_CV2
THEANO_FLAGS=device=gpu,floatX=float32 python train6.py 3 > output_6_CV3
THEANO_FLAGS=device=gpu,floatX=float32 python train6.py 4 > output_6_CV4
THEANO_FLAGS=device=gpu,floatX=float32 python train6.py 5 > output_6_CV5
THEANO_FLAGS=device=gpu,floatX=float32 python train6.py 6 > output_6_CV6
THEANO_FLAGS=device=gpu,floatX=float32 python train6.py 7 > output_6_CV7
THEANO_FLAGS=device=gpu,floatX=float32 python train6.py 8 > output_6_CV8
THEANO_FLAGS=device=gpu,floatX=float32 python train6.py 9 > output_6_CV9
THEANO_FLAGS=device=gpu,floatX=float32 python train6.py 10 > output_6_CV10
