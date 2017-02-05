#!/bin/bash

THEANO_FLAGS=device=gpu,floatX=float32 python train1.py 1 > output_1_CV1
THEANO_FLAGS=device=gpu,floatX=float32 python train1.py 2 > output_1_CV2
THEANO_FLAGS=device=gpu,floatX=float32 python train1.py 3 > output_1_CV3
THEANO_FLAGS=device=gpu,floatX=float32 python train1.py 4 > output_1_CV4
THEANO_FLAGS=device=gpu,floatX=float32 python train1.py 5 > output_1_CV5
THEANO_FLAGS=device=gpu,floatX=float32 python train1.py 6 > output_1_CV6
THEANO_FLAGS=device=gpu,floatX=float32 python train1.py 7 > output_1_CV7
THEANO_FLAGS=device=gpu,floatX=float32 python train1.py 8 > output_1_CV8
THEANO_FLAGS=device=gpu,floatX=float32 python train1.py 9 > output_1_CV9
THEANO_FLAGS=device=gpu,floatX=float32 python train1.py 10 > output_1_CV10
