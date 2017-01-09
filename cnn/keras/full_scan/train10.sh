#!/bin/bash

THEANO_FLAGS=device=gpu,floatX=float32 python train10.py 1 > output_10_CV1
THEANO_FLAGS=device=gpu,floatX=float32 python train10.py 2 > output_10_CV2
THEANO_FLAGS=device=gpu,floatX=float32 python train10.py 3 > output_10_CV3
THEANO_FLAGS=device=gpu,floatX=float32 python train10.py 4 > output_10_CV4
THEANO_FLAGS=device=gpu,floatX=float32 python train10.py 5 > output_10_CV5
THEANO_FLAGS=device=gpu,floatX=float32 python train10.py 6 > output_10_CV6
THEANO_FLAGS=device=gpu,floatX=float32 python train10.py 7 > output_10_CV7
THEANO_FLAGS=device=gpu,floatX=float32 python train10.py 8 > output_10_CV8
THEANO_FLAGS=device=gpu,floatX=float32 python train10.py 9 > output_10_CV9
THEANO_FLAGS=device=gpu,floatX=float32 python train10.py 10 > output_10_CV10
