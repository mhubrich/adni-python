#!/bin/bash

THEANO_FLAGS=device=gpu,floatX=float32 python train_adadelta11.py 1 > output_adadelta_test12_CV1
THEANO_FLAGS=device=gpu,floatX=float32 python train_adadelta11.py 2 > output_adadelta_test12_CV2
THEANO_FLAGS=device=gpu,floatX=float32 python train_adadelta11.py 3 > output_adadelta_test12_CV3
THEANO_FLAGS=device=gpu,floatX=float32 python train_adadelta11.py 4 > output_adadelta_test12_CV4
THEANO_FLAGS=device=gpu,floatX=float32 python train_adadelta11.py 5 > output_adadelta_test12_CV5
THEANO_FLAGS=device=gpu,floatX=float32 python train_adadelta11.py 6 > output_adadelta_test12_CV6
THEANO_FLAGS=device=gpu,floatX=float32 python train_adadelta11.py 7 > output_adadelta_test12_CV7
THEANO_FLAGS=device=gpu,floatX=float32 python train_adadelta11.py 8 > output_adadelta_test12_CV8
THEANO_FLAGS=device=gpu,floatX=float32 python train_adadelta11.py 9 > output_adadelta_test12_CV9
THEANO_FLAGS=device=gpu,floatX=float32 python train_adadelta11.py 10 > output_adadelta_test12_CV10
