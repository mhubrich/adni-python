#!/bin/bash

THEANO_FLAGS=device=gpu,floatX=float32 python train12.py 1 > output_12_CV1
THEANO_FLAGS=device=gpu,floatX=float32 python train12.py 2 > output_12_CV2
THEANO_FLAGS=device=gpu,floatX=float32 python train12.py 3 > output_12_CV3
THEANO_FLAGS=device=gpu,floatX=float32 python train12.py 4 > output_12_CV4
THEANO_FLAGS=device=gpu,floatX=float32 python train12.py 5 > output_12_CV5
THEANO_FLAGS=device=gpu,floatX=float32 python train12.py 6 > output_12_CV6
THEANO_FLAGS=device=gpu,floatX=float32 python train12.py 7 > output_12_CV7
THEANO_FLAGS=device=gpu,floatX=float32 python train12.py 8 > output_12_CV8
THEANO_FLAGS=device=gpu,floatX=float32 python train12.py 9 > output_12_CV9
THEANO_FLAGS=device=gpu,floatX=float32 python train12.py 10 > output_12_CV10
