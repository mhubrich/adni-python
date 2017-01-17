#!/bin/bash

THEANO_FLAGS=device=gpu,floatX=float32 python train19.py 1 > output_19_CV1
THEANO_FLAGS=device=gpu,floatX=float32 python train19.py 2 > output_19_CV2
THEANO_FLAGS=device=gpu,floatX=float32 python train19.py 3 > output_19_CV3
THEANO_FLAGS=device=gpu,floatX=float32 python train19.py 4 > output_19_CV4
THEANO_FLAGS=device=gpu,floatX=float32 python train19.py 5 > output_19_CV5
THEANO_FLAGS=device=gpu,floatX=float32 python train19.py 6 > output_19_CV6
THEANO_FLAGS=device=gpu,floatX=float32 python train19.py 7 > output_19_CV7
THEANO_FLAGS=device=gpu,floatX=float32 python train19.py 8 > output_19_CV8
THEANO_FLAGS=device=gpu,floatX=float32 python train19.py 9 > output_19_CV9
THEANO_FLAGS=device=gpu,floatX=float32 python train19.py 10 > output_19_CV10
