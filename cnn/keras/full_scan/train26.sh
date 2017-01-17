#!/bin/bash

THEANO_FLAGS=device=gpu,floatX=float32 python train26.py 1 > output_26_CV1
THEANO_FLAGS=device=gpu,floatX=float32 python train26.py 2 > output_26_CV2
THEANO_FLAGS=device=gpu,floatX=float32 python train26.py 3 > output_26_CV3
THEANO_FLAGS=device=gpu,floatX=float32 python train26.py 4 > output_26_CV4
THEANO_FLAGS=device=gpu,floatX=float32 python train26.py 5 > output_26_CV5
THEANO_FLAGS=device=gpu,floatX=float32 python train26.py 6 > output_26_CV6
THEANO_FLAGS=device=gpu,floatX=float32 python train26.py 7 > output_26_CV7
THEANO_FLAGS=device=gpu,floatX=float32 python train26.py 8 > output_26_CV8
THEANO_FLAGS=device=gpu,floatX=float32 python train26.py 9 > output_26_CV9
THEANO_FLAGS=device=gpu,floatX=float32 python train26.py 10 > output_26_CV10
