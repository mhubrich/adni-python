#!/bin/bash

THEANO_FLAGS=device=gpu,floatX=float32 python train31.py 1 > output_31_CV1
THEANO_FLAGS=device=gpu,floatX=float32 python train31.py 2 > output_31_CV2
THEANO_FLAGS=device=gpu,floatX=float32 python train31.py 3 > output_31_CV3
THEANO_FLAGS=device=gpu,floatX=float32 python train31.py 4 > output_31_CV4
THEANO_FLAGS=device=gpu,floatX=float32 python train31.py 5 > output_31_CV5
THEANO_FLAGS=device=gpu,floatX=float32 python train31.py 6 > output_31_CV6
THEANO_FLAGS=device=gpu,floatX=float32 python train31.py 7 > output_31_CV7
THEANO_FLAGS=device=gpu,floatX=float32 python train31.py 8 > output_31_CV8
THEANO_FLAGS=device=gpu,floatX=float32 python train31.py 9 > output_31_CV9
THEANO_FLAGS=device=gpu,floatX=float32 python train31.py 10 > output_31_CV10
