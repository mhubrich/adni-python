#!/bin/bash

THEANO_FLAGS=device=gpu,floatX=float32 python train27.py 1 > output_27_CV1
THEANO_FLAGS=device=gpu,floatX=float32 python train27.py 2 > output_27_CV2
THEANO_FLAGS=device=gpu,floatX=float32 python train27.py 3 > output_27_CV3
THEANO_FLAGS=device=gpu,floatX=float32 python train27.py 4 > output_27_CV4
THEANO_FLAGS=device=gpu,floatX=float32 python train27.py 5 > output_27_CV5
THEANO_FLAGS=device=gpu,floatX=float32 python train27.py 6 > output_27_CV6
THEANO_FLAGS=device=gpu,floatX=float32 python train27.py 7 > output_27_CV7
THEANO_FLAGS=device=gpu,floatX=float32 python train27.py 8 > output_27_CV8
THEANO_FLAGS=device=gpu,floatX=float32 python train27.py 9 > output_27_CV9
THEANO_FLAGS=device=gpu,floatX=float32 python train27.py 10 > output_27_CV10
