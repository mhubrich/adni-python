#!/bin/bash

THEANO_FLAGS=device=gpu,floatX=float32 python train24.py 1 > output_24_CV1
THEANO_FLAGS=device=gpu,floatX=float32 python train24.py 2 > output_24_CV2
THEANO_FLAGS=device=gpu,floatX=float32 python train24.py 3 > output_24_CV3
THEANO_FLAGS=device=gpu,floatX=float32 python train24.py 4 > output_24_CV4
THEANO_FLAGS=device=gpu,floatX=float32 python train24.py 5 > output_24_CV5
THEANO_FLAGS=device=gpu,floatX=float32 python train24.py 6 > output_24_CV6
THEANO_FLAGS=device=gpu,floatX=float32 python train24.py 7 > output_24_CV7
THEANO_FLAGS=device=gpu,floatX=float32 python train24.py 8 > output_24_CV8
THEANO_FLAGS=device=gpu,floatX=float32 python train24.py 9 > output_24_CV9
THEANO_FLAGS=device=gpu,floatX=float32 python train24.py 10 > output_24_CV10
