#!/bin/bash

THEANO_FLAGS=device=gpu,floatX=float32 python train30.py 1 > output_30_CV1
THEANO_FLAGS=device=gpu,floatX=float32 python train30.py 2 > output_30_CV2
THEANO_FLAGS=device=gpu,floatX=float32 python train30.py 3 > output_30_CV3
THEANO_FLAGS=device=gpu,floatX=float32 python train30.py 4 > output_30_CV4
THEANO_FLAGS=device=gpu,floatX=float32 python train30.py 5 > output_30_CV5
THEANO_FLAGS=device=gpu,floatX=float32 python train30.py 6 > output_30_CV6
THEANO_FLAGS=device=gpu,floatX=float32 python train30.py 7 > output_30_CV7
THEANO_FLAGS=device=gpu,floatX=float32 python train30.py 8 > output_30_CV8
THEANO_FLAGS=device=gpu,floatX=float32 python train30.py 9 > output_30_CV9
THEANO_FLAGS=device=gpu,floatX=float32 python train30.py 10 > output_30_CV10
