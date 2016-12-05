#!/bin/bash

THEANO_FLAGS=device=gpu,floatX=float32 python train_adadelta4.py 1 > output_adadelta_testset_CV1
THEANO_FLAGS=device=gpu,floatX=float32 python train_adadelta4.py 2 > output_adadelta_testset_CV2
THEANO_FLAGS=device=gpu,floatX=float32 python train_adadelta4.py 3 > output_adadelta_testset_CV3
THEANO_FLAGS=device=gpu,floatX=float32 python train_adadelta4.py 4 > output_adadelta_testset_CV4
THEANO_FLAGS=device=gpu,floatX=float32 python train_adadelta4.py 5 > output_adadelta_testset_CV5
THEANO_FLAGS=device=gpu,floatX=float32 python train_adadelta4.py 6 > output_adadelta_testset_CV6
THEANO_FLAGS=device=gpu,floatX=float32 python train_adadelta4.py 7 > output_adadelta_testset_CV7
THEANO_FLAGS=device=gpu,floatX=float32 python train_adadelta4.py 8 > output_adadelta_testset_CV8
THEANO_FLAGS=device=gpu,floatX=float32 python train_adadelta4.py 9 > output_adadelta_testset_CV9
THEANO_FLAGS=device=gpu,floatX=float32 python train_adadelta4.py 10 > output_adadelta_testset_CV10
