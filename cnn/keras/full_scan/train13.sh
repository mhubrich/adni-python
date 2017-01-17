#!/bin/bash

THEANO_FLAGS=device=gpu,floatX=float32 python train13.py 1 > output_13_CV1
THEANO_FLAGS=device=gpu,floatX=float32 python train13.py 2 > output_13_CV2
THEANO_FLAGS=device=gpu,floatX=float32 python train13.py 3 > output_13_CV3
THEANO_FLAGS=device=gpu,floatX=float32 python train13.py 4 > output_13_CV4
THEANO_FLAGS=device=gpu,floatX=float32 python train13.py 5 > output_13_CV5
THEANO_FLAGS=device=gpu,floatX=float32 python train13.py 6 > output_13_CV6
THEANO_FLAGS=device=gpu,floatX=float32 python train13.py 7 > output_13_CV7
THEANO_FLAGS=device=gpu,floatX=float32 python train13.py 8 > output_13_CV8
THEANO_FLAGS=device=gpu,floatX=float32 python train13.py 9 > output_13_CV9
THEANO_FLAGS=device=gpu,floatX=float32 python train13.py 10 > output_13_CV10
