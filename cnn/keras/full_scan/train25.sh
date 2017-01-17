#!/bin/bash

THEANO_FLAGS=device=gpu,floatX=float32 python train25.py 1 > output_25_CV1
THEANO_FLAGS=device=gpu,floatX=float32 python train25.py 2 > output_25_CV2
THEANO_FLAGS=device=gpu,floatX=float32 python train25.py 3 > output_25_CV3
THEANO_FLAGS=device=gpu,floatX=float32 python train25.py 4 > output_25_CV4
THEANO_FLAGS=device=gpu,floatX=float32 python train25.py 5 > output_25_CV5
THEANO_FLAGS=device=gpu,floatX=float32 python train25.py 6 > output_25_CV6
THEANO_FLAGS=device=gpu,floatX=float32 python train25.py 7 > output_25_CV7
THEANO_FLAGS=device=gpu,floatX=float32 python train25.py 8 > output_25_CV8
THEANO_FLAGS=device=gpu,floatX=float32 python train25.py 9 > output_25_CV9
THEANO_FLAGS=device=gpu,floatX=float32 python train25.py 10 > output_25_CV10
