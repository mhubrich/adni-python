#!/bin/bash

THEANO_FLAGS=device=gpu,floatX=float32 python train16.py 1 > output_16_CV1
THEANO_FLAGS=device=gpu,floatX=float32 python train16.py 2 > output_16_CV2
THEANO_FLAGS=device=gpu,floatX=float32 python train16.py 3 > output_16_CV3
THEANO_FLAGS=device=gpu,floatX=float32 python train16.py 4 > output_16_CV4
THEANO_FLAGS=device=gpu,floatX=float32 python train16.py 5 > output_16_CV5
THEANO_FLAGS=device=gpu,floatX=float32 python train16.py 6 > output_16_CV6
THEANO_FLAGS=device=gpu,floatX=float32 python train16.py 7 > output_16_CV7
THEANO_FLAGS=device=gpu,floatX=float32 python train16.py 8 > output_16_CV8
THEANO_FLAGS=device=gpu,floatX=float32 python train16.py 9 > output_16_CV9
THEANO_FLAGS=device=gpu,floatX=float32 python train16.py 10 > output_16_CV10
