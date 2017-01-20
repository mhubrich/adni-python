#!/bin/bash

THEANO_FLAGS=device=gpu,floatX=float32,lib.cnmem=1.0 python train2conv.py 1 > output_conv2_CV1
THEANO_FLAGS=device=gpu,floatX=float32,lib.cnmem=1.0 python train2conv.py 2 > output_conv2_CV2
THEANO_FLAGS=device=gpu,floatX=float32,lib.cnmem=1.0 python train2conv.py 3 > output_conv2_CV3
THEANO_FLAGS=device=gpu,floatX=float32,lib.cnmem=1.0 python train2conv.py 4 > output_conv2_CV4
THEANO_FLAGS=device=gpu,floatX=float32,lib.cnmem=1.0 python train2conv.py 5 > output_conv2_CV5
THEANO_FLAGS=device=gpu,floatX=float32,lib.cnmem=1.0 python train2conv.py 6 > output_conv2_CV6
THEANO_FLAGS=device=gpu,floatX=float32,lib.cnmem=1.0 python train2conv.py 7 > output_conv2_CV7
THEANO_FLAGS=device=gpu,floatX=float32,lib.cnmem=1.0 python train2conv.py 8 > output_conv2_CV8
THEANO_FLAGS=device=gpu,floatX=float32,lib.cnmem=1.0 python train2conv.py 9 > output_conv2_CV9
THEANO_FLAGS=device=gpu,floatX=float32,lib.cnmem=1.0 python train2conv.py 10 > output_conv2_CV10
