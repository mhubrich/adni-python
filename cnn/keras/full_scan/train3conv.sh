#!/bin/bash

THEANO_FLAGS=device=gpu,floatX=float32 python train3conv.py 1 > output_conv3_CV1
THEANO_FLAGS=device=gpu,floatX=float32 python train3conv.py 2 > output_conv3_CV2
THEANO_FLAGS=device=gpu,floatX=float32 python train3conv.py 3 > output_conv3_CV3
THEANO_FLAGS=device=gpu,floatX=float32 python train3conv.py 4 > output_conv3_CV4
THEANO_FLAGS=device=gpu,floatX=float32 python train3conv.py 5 > output_conv3_CV5
THEANO_FLAGS=device=gpu,floatX=float32 python train3conv.py 6 > output_conv3_CV6
THEANO_FLAGS=device=gpu,floatX=float32 python train3conv.py 7 > output_conv3_CV7
THEANO_FLAGS=device=gpu,floatX=float32 python train3conv.py 8 > output_conv3_CV8
THEANO_FLAGS=device=gpu,floatX=float32 python train3conv.py 9 > output_conv3_CV9
THEANO_FLAGS=device=gpu,floatX=float32 python train3conv.py 10 > output_conv3_CV10
