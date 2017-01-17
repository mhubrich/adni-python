#!/bin/bash

THEANO_FLAGS=device=gpu,floatX=float32 python train21.py 1 > output_21_CV1
THEANO_FLAGS=device=gpu,floatX=float32 python train21.py 2 > output_21_CV2
THEANO_FLAGS=device=gpu,floatX=float32 python train21.py 3 > output_21_CV3
THEANO_FLAGS=device=gpu,floatX=float32 python train21.py 4 > output_21_CV4
THEANO_FLAGS=device=gpu,floatX=float32 python train21.py 5 > output_21_CV5
THEANO_FLAGS=device=gpu,floatX=float32 python train21.py 6 > output_21_CV6
THEANO_FLAGS=device=gpu,floatX=float32 python train21.py 7 > output_21_CV7
THEANO_FLAGS=device=gpu,floatX=float32 python train21.py 8 > output_21_CV8
THEANO_FLAGS=device=gpu,floatX=float32 python train21.py 9 > output_21_CV9
THEANO_FLAGS=device=gpu,floatX=float32 python train21.py 10 > output_21_CV10
