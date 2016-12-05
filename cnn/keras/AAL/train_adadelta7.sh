#!/bin/bash

THEANO_FLAGS=device=gpu,floatX=float32 python train_adadelta7.py 1 > output_adadelta_test7_CV1
THEANO_FLAGS=device=gpu,floatX=float32 python train_adadelta7.py 2 > output_adadelta_test7_CV2
THEANO_FLAGS=device=gpu,floatX=float32 python train_adadelta7.py 3 > output_adadelta_test7_CV3
THEANO_FLAGS=device=gpu,floatX=float32 python train_adadelta7.py 4 > output_adadelta_test7_CV4
THEANO_FLAGS=device=gpu,floatX=float32 python train_adadelta7.py 5 > output_adadelta_test7_CV5
THEANO_FLAGS=device=gpu,floatX=float32 python train_adadelta7.py 6 > output_adadelta_test7_CV6
THEANO_FLAGS=device=gpu,floatX=float32 python train_adadelta7.py 7 > output_adadelta_test7_CV7
THEANO_FLAGS=device=gpu,floatX=float32 python train_adadelta7.py 8 > output_adadelta_test7_CV8
THEANO_FLAGS=device=gpu,floatX=float32 python train_adadelta7.py 9 > output_adadelta_test7_CV9
THEANO_FLAGS=device=gpu,floatX=float32 python train_adadelta7.py 10 > output_adadelta_test7_CV10
