#!/bin/bash

THEANO_FLAGS=device=gpu,floatX=float32 python train_sigmoid.py 1 > output_hinge_CV1
THEANO_FLAGS=device=gpu,floatX=float32 python train_sigmoid.py 2 > output_hinge_CV2
THEANO_FLAGS=device=gpu,floatX=float32 python train_sigmoid.py 3 > output_hinge_CV3
THEANO_FLAGS=device=gpu,floatX=float32 python train_sigmoid.py 4 > output_hinge_CV4
THEANO_FLAGS=device=gpu,floatX=float32 python train_sigmoid.py 5 > output_hinge_CV5
THEANO_FLAGS=device=gpu,floatX=float32 python train_sigmoid.py 6 > output_hinge_CV6
THEANO_FLAGS=device=gpu,floatX=float32 python train_sigmoid.py 7 > output_hinge_CV7
THEANO_FLAGS=device=gpu,floatX=float32 python train_sigmoid.py 8 > output_hinge_CV8
THEANO_FLAGS=device=gpu,floatX=float32 python train_sigmoid.py 9 > output_hinge_CV9
THEANO_FLAGS=device=gpu,floatX=float32 python train_sigmoid.py 10 > output_hinge_CV10
