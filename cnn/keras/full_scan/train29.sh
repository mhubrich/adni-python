#!/bin/bash

THEANO_FLAGS=device=gpu,floatX=float32 python train29.py 1 > output_29_CV1
THEANO_FLAGS=device=gpu,floatX=float32 python train29.py 2 > output_29_CV2
THEANO_FLAGS=device=gpu,floatX=float32 python train29.py 3 > output_29_CV3
THEANO_FLAGS=device=gpu,floatX=float32 python train29.py 4 > output_29_CV4
THEANO_FLAGS=device=gpu,floatX=float32 python train29.py 5 > output_29_CV5
THEANO_FLAGS=device=gpu,floatX=float32 python train29.py 6 > output_29_CV6
THEANO_FLAGS=device=gpu,floatX=float32 python train29.py 7 > output_29_CV7
THEANO_FLAGS=device=gpu,floatX=float32 python train29.py 8 > output_29_CV8
THEANO_FLAGS=device=gpu,floatX=float32 python train29.py 9 > output_29_CV9
THEANO_FLAGS=device=gpu,floatX=float32 python train29.py 10 > output_29_CV10
