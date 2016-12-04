#!/bin/bash

THEANO_FLAGS=device=gpu,floatX=float32 python train_adadelta4.py 1 > output_adadelta_test4_CV1
THEANO_FLAGS=device=gpu,floatX=float32 python train_adadelta4.py 2 > output_adadelta_test4_CV2
THEANO_FLAGS=device=gpu,floatX=float32 python train_adadelta4.py 3 > output_adadelta_test4_CV3
THEANO_FLAGS=device=gpu,floatX=float32 python train_adadelta4.py 4 > output_adadelta_test4_CV4
THEANO_FLAGS=device=gpu,floatX=float32 python train_adadelta4.py 5 > output_adadelta_test4_CV5
THEANO_FLAGS=device=gpu,floatX=float32 python train_adadelta4.py 6 > output_adadelta_test4_CV6
THEANO_FLAGS=device=gpu,floatX=float32 python train_adadelta4.py 7 > output_adadelta_test4_CV7
THEANO_FLAGS=device=gpu,floatX=float32 python train_adadelta4.py 8 > output_adadelta_test4_CV8
THEANO_FLAGS=device=gpu,floatX=float32 python train_adadelta4.py 9 > output_adadelta_test4_CV9
THEANO_FLAGS=device=gpu,floatX=float32 python train_adadelta4.py 10 > output_adadelta_test4_CV10
