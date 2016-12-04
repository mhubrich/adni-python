#!/bin/bash

THEANO_FLAGS=device=gpu,floatX=float32 python train_adadelta.py 1 > output_adadelta_test2_CV1
THEANO_FLAGS=device=gpu,floatX=float32 python train_adadelta.py 2 > output_adadelta_test2_CV2
THEANO_FLAGS=device=gpu,floatX=float32 python train_adadelta.py 3 > output_adadelta_test2_CV3
THEANO_FLAGS=device=gpu,floatX=float32 python train_adadelta.py 4 > output_adadelta_test2_CV4
THEANO_FLAGS=device=gpu,floatX=float32 python train_adadelta.py 5 > output_adadelta_test2_CV5
THEANO_FLAGS=device=gpu,floatX=float32 python train_adadelta.py 6 > output_adadelta_test2_CV6
THEANO_FLAGS=device=gpu,floatX=float32 python train_adadelta.py 7 > output_adadelta_test2_CV7
THEANO_FLAGS=device=gpu,floatX=float32 python train_adadelta.py 8 > output_adadelta_test2_CV8
THEANO_FLAGS=device=gpu,floatX=float32 python train_adadelta.py 9 > output_adadelta_test2_CV9
THEANO_FLAGS=device=gpu,floatX=float32 python train_adadelta.py 10 > output_adadelta_test2_CV10
