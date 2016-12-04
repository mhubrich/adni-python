#!/bin/bash

THEANO_FLAGS=device=gpu,floatX=float32 python train_nadam.py 1 > output_nadam_test_CV1
THEANO_FLAGS=device=gpu,floatX=float32 python train_nadam.py 2 > output_nadam_test_CV2
THEANO_FLAGS=device=gpu,floatX=float32 python train_nadam.py 3 > output_nadam_test_CV3
THEANO_FLAGS=device=gpu,floatX=float32 python train_nadam.py 4 > output_nadam_test_CV4
THEANO_FLAGS=device=gpu,floatX=float32 python train_nadam.py 5 > output_nadam_test_CV5
THEANO_FLAGS=device=gpu,floatX=float32 python train_nadam.py 6 > output_nadam_test_CV6
THEANO_FLAGS=device=gpu,floatX=float32 python train_nadam.py 7 > output_nadam_test_CV7
THEANO_FLAGS=device=gpu,floatX=float32 python train_nadam.py 8 > output_nadam_test_CV8
THEANO_FLAGS=device=gpu,floatX=float32 python train_nadam.py 9 > output_nadam_test_CV9
THEANO_FLAGS=device=gpu,floatX=float32 python train_nadam.py 10 > output_nadam_test_CV10
