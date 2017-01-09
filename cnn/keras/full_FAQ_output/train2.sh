#!/bin/bash

THEANO_FLAGS=device=gpu,floatX=float32 python train.py 6 > output_FAQ_out_1_CV6
THEANO_FLAGS=device=gpu,floatX=float32 python train.py 7 > output_FAQ_out_1_CV7
THEANO_FLAGS=device=gpu,floatX=float32 python train.py 8 > output_FAQ_out_1_CV8
THEANO_FLAGS=device=gpu,floatX=float32 python train.py 9 > output_FAQ_out_1_CV9
THEANO_FLAGS=device=gpu,floatX=float32 python train.py 10 > output_FAQ_out_1_CV10
