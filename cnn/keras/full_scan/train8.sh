#!/bin/bash

THEANO_FLAGS=device=gpu,floatX=float32 python train8.py 1 > output_8_CV1
THEANO_FLAGS=device=gpu,floatX=float32 python train8.py 2 > output_8_CV2
THEANO_FLAGS=device=gpu,floatX=float32 python train8.py 3 > output_8_CV3
THEANO_FLAGS=device=gpu,floatX=float32 python train8.py 4 > output_8_CV4
THEANO_FLAGS=device=gpu,floatX=float32 python train8.py 5 > output_8_CV5
THEANO_FLAGS=device=gpu,floatX=float32 python train8.py 6 > output_8_CV6
THEANO_FLAGS=device=gpu,floatX=float32 python train8.py 7 > output_8_CV7
THEANO_FLAGS=device=gpu,floatX=float32 python train8.py 8 > output_8_CV8
THEANO_FLAGS=device=gpu,floatX=float32 python train8.py 9 > output_8_CV9
THEANO_FLAGS=device=gpu,floatX=float32 python train8.py 10 > output_8_CV10
