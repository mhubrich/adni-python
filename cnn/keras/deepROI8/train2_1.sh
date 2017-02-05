#!/bin/bash

THEANO_FLAGS=device=gpu,floatX=float32,lib.cnmem=1.0 python train2.py 1 > output_2_CV1
THEANO_FLAGS=device=gpu,floatX=float32,lib.cnmem=1.0 python train2.py 2 > output_2_CV2
THEANO_FLAGS=device=gpu,floatX=float32,lib.cnmem=1.0 python train2.py 3 > output_2_CV3
THEANO_FLAGS=device=gpu,floatX=float32,lib.cnmem=1.0 python train2.py 4 > output_2_CV4
THEANO_FLAGS=device=gpu,floatX=float32,lib.cnmem=1.0 python train2.py 5 > output_2_CV5
