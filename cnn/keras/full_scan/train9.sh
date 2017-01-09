#!/bin/bash

THEANO_FLAGS=device=gpu,floatX=float32 python train9.py 1 > output_9_CV1
THEANO_FLAGS=device=gpu,floatX=float32 python train9.py 2 > output_9_CV2
THEANO_FLAGS=device=gpu,floatX=float32 python train9.py 3 > output_9_CV3
THEANO_FLAGS=device=gpu,floatX=float32 python train9.py 4 > output_9_CV4
THEANO_FLAGS=device=gpu,floatX=float32 python train9.py 5 > output_9_CV5
THEANO_FLAGS=device=gpu,floatX=float32 python train9.py 6 > output_9_CV6
THEANO_FLAGS=device=gpu,floatX=float32 python train9.py 7 > output_9_CV7
THEANO_FLAGS=device=gpu,floatX=float32 python train9.py 8 > output_9_CV8
THEANO_FLAGS=device=gpu,floatX=float32 python train9.py 9 > output_9_CV9
THEANO_FLAGS=device=gpu,floatX=float32 python train9.py 10 > output_9_CV10
