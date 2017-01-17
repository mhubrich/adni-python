#!/bin/bash

THEANO_FLAGS=device=gpu,floatX=float32 python train11.py 1 > output_11_CV1
THEANO_FLAGS=device=gpu,floatX=float32 python train11.py 2 > output_11_CV2
THEANO_FLAGS=device=gpu,floatX=float32 python train11.py 3 > output_11_CV3
THEANO_FLAGS=device=gpu,floatX=float32 python train11.py 4 > output_11_CV4
THEANO_FLAGS=device=gpu,floatX=float32 python train11.py 5 > output_11_CV5
THEANO_FLAGS=device=gpu,floatX=float32 python train11.py 6 > output_11_CV6
THEANO_FLAGS=device=gpu,floatX=float32 python train11.py 7 > output_11_CV7
THEANO_FLAGS=device=gpu,floatX=float32 python train11.py 8 > output_11_CV8
THEANO_FLAGS=device=gpu,floatX=float32 python train11.py 9 > output_11_CV9
THEANO_FLAGS=device=gpu,floatX=float32 python train11.py 10 > output_11_CV10
