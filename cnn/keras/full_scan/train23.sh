#!/bin/bash

THEANO_FLAGS=device=gpu,floatX=float32 python train23.py 1 > output_23_CV1
THEANO_FLAGS=device=gpu,floatX=float32 python train23.py 2 > output_23_CV2
THEANO_FLAGS=device=gpu,floatX=float32 python train23.py 3 > output_23_CV3
THEANO_FLAGS=device=gpu,floatX=float32 python train23.py 4 > output_23_CV4
THEANO_FLAGS=device=gpu,floatX=float32 python train23.py 5 > output_23_CV5
THEANO_FLAGS=device=gpu,floatX=float32 python train23.py 6 > output_23_CV6
THEANO_FLAGS=device=gpu,floatX=float32 python train23.py 7 > output_23_CV7
THEANO_FLAGS=device=gpu,floatX=float32 python train23.py 8 > output_23_CV8
THEANO_FLAGS=device=gpu,floatX=float32 python train23.py 9 > output_23_CV9
THEANO_FLAGS=device=gpu,floatX=float32 python train23.py 10 > output_23_CV10
