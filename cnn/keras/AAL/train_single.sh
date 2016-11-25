#!/bin/bash

THEANO_FLAGS=device=gpu,floatX=float32 python train_single.py 1 > output_AAL61_new_2_CV1
THEANO_FLAGS=device=gpu,floatX=float32 python train_single.py 2 > output_AAL61_new_2_CV2
THEANO_FLAGS=device=gpu,floatX=float32 python train_single.py 3 > output_AAL61_new_2_CV3
THEANO_FLAGS=device=gpu,floatX=float32 python train_single.py 4 > output_AAL61_new_2_CV4
THEANO_FLAGS=device=gpu,floatX=float32 python train_single.py 5 > output_AAL61_new_2_CV5
THEANO_FLAGS=device=gpu,floatX=float32 python train_single.py 6 > output_AAL61_new_2_CV6
THEANO_FLAGS=device=gpu,floatX=float32 python train_single.py 7 > output_AAL61_new_2_CV7
THEANO_FLAGS=device=gpu,floatX=float32 python train_single.py 8 > output_AAL61_new_2_CV8
THEANO_FLAGS=device=gpu,floatX=float32 python train_single.py 9 > output_AAL61_new_2_CV9
THEANO_FLAGS=device=gpu,floatX=float32 python train_single.py 10 > output_AAL61_new_2_CV10

