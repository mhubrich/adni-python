#!/bin/bash

THEANO_FLAGS=device=gpu,floatX=float32 python train1conv.py 1 > output_conv1_CV1
THEANO_FLAGS=device=gpu,floatX=float32 python train1conv.py 2 > output_conv1_CV2
THEANO_FLAGS=device=gpu,floatX=float32 python train1conv.py 3 > output_conv1_CV3
THEANO_FLAGS=device=gpu,floatX=float32 python train1conv.py 4 > output_conv1_CV4
THEANO_FLAGS=device=gpu,floatX=float32 python train1conv.py 5 > output_conv1_CV5
THEANO_FLAGS=device=gpu,floatX=float32 python train1conv.py 6 > output_conv1_CV6
THEANO_FLAGS=device=gpu,floatX=float32 python train1conv.py 7 > output_conv1_CV7
THEANO_FLAGS=device=gpu,floatX=float32 python train1conv.py 8 > output_conv1_CV8
THEANO_FLAGS=device=gpu,floatX=float32 python train1conv.py 9 > output_conv1_CV9
THEANO_FLAGS=device=gpu,floatX=float32 python train1conv.py 10 > output_conv1_CV10
