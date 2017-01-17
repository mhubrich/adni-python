#!/bin/bash

THEANO_FLAGS=device=gpu,floatX=float32 python train18.py 1 > output_18_CV1
THEANO_FLAGS=device=gpu,floatX=float32 python train18.py 2 > output_18_CV2
THEANO_FLAGS=device=gpu,floatX=float32 python train18.py 3 > output_18_CV3
THEANO_FLAGS=device=gpu,floatX=float32 python train18.py 4 > output_18_CV4
THEANO_FLAGS=device=gpu,floatX=float32 python train18.py 5 > output_18_CV5
THEANO_FLAGS=device=gpu,floatX=float32 python train18.py 6 > output_18_CV6
THEANO_FLAGS=device=gpu,floatX=float32 python train18.py 7 > output_18_CV7
THEANO_FLAGS=device=gpu,floatX=float32 python train18.py 8 > output_18_CV8
THEANO_FLAGS=device=gpu,floatX=float32 python train18.py 9 > output_18_CV9
THEANO_FLAGS=device=gpu,floatX=float32 python train18.py 10 > output_18_CV10
