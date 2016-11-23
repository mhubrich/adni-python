#!/bin/bash

THEANO_FLAGS=device=gpu,floatX=float32 python train_balanced.py 1 > output_first_balanced_CV1
THEANO_FLAGS=device=gpu,floatX=float32 python train_balanced.py 2 > output_first_balanced_CV2
THEANO_FLAGS=device=gpu,floatX=float32 python train_balanced.py 3 > output_first_balanced_CV3
THEANO_FLAGS=device=gpu,floatX=float32 python train_balanced.py 4 > output_first_balanced_CV4
THEANO_FLAGS=device=gpu,floatX=float32 python train_balanced.py 5 > output_first_balanced_CV5
THEANO_FLAGS=device=gpu,floatX=float32 python train_balanced.py 6 > output_first_balanced_CV6
THEANO_FLAGS=device=gpu,floatX=float32 python train_balanced.py 7 > output_first_balanced_CV7
THEANO_FLAGS=device=gpu,floatX=float32 python train_balanced.py 8 > output_first_balanced_CV8
THEANO_FLAGS=device=gpu,floatX=float32 python train_balanced.py 9 > output_first_balanced_CV9
THEANO_FLAGS=device=gpu,floatX=float32 python train_balanced.py 10 > output_first_balanced_CV10

