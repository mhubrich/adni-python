#!/bin/bash

THEANO_FLAGS=device=gpu,floatX=float32 python train_weights.py 1 > output_first_weights_CV1
THEANO_FLAGS=device=gpu,floatX=float32 python train_weights.py 2 > output_first_weights_CV2
THEANO_FLAGS=device=gpu,floatX=float32 python train_weights.py 3 > output_first_weights_CV3
THEANO_FLAGS=device=gpu,floatX=float32 python train_weights.py 4 > output_first_weights_CV4
THEANO_FLAGS=device=gpu,floatX=float32 python train_weights.py 5 > output_first_weights_CV5
THEANO_FLAGS=device=gpu,floatX=float32 python train_weights.py 6 > output_first_weights_CV6
THEANO_FLAGS=device=gpu,floatX=float32 python train_weights.py 7 > output_first_weights_CV7
THEANO_FLAGS=device=gpu,floatX=float32 python train_weights.py 8 > output_first_weights_CV8
THEANO_FLAGS=device=gpu,floatX=float32 python train_weights.py 9 > output_first_weights_CV9
THEANO_FLAGS=device=gpu,floatX=float32 python train_weights.py 10 > output_first_weights_CV10

