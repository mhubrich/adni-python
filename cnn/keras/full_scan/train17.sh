#!/bin/bash

THEANO_FLAGS=device=gpu,floatX=float32 python train17.py 1 > output_17_CV1
THEANO_FLAGS=device=gpu,floatX=float32 python train17.py 2 > output_17_CV2
THEANO_FLAGS=device=gpu,floatX=float32 python train17.py 3 > output_17_CV3
THEANO_FLAGS=device=gpu,floatX=float32 python train17.py 4 > output_17_CV4
THEANO_FLAGS=device=gpu,floatX=float32 python train17.py 5 > output_17_CV5
THEANO_FLAGS=device=gpu,floatX=float32 python train17.py 6 > output_17_CV6
THEANO_FLAGS=device=gpu,floatX=float32 python train17.py 7 > output_17_CV7
THEANO_FLAGS=device=gpu,floatX=float32 python train17.py 8 > output_17_CV8
THEANO_FLAGS=device=gpu,floatX=float32 python train17.py 9 > output_17_CV9
THEANO_FLAGS=device=gpu,floatX=float32 python train17.py 10 > output_17_CV10
