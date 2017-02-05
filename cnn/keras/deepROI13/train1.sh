#!/bin/bash

THEANO_FLAGS=device=gpu,floatX=float32,lib.cnmem=0.2 python train1.py 1 > output_1_CV1
THEANO_FLAGS=device=gpu,floatX=float32,lib.cnmem=0.2 python train1.py 2 > output_1_CV2
THEANO_FLAGS=device=gpu,floatX=float32,lib.cnmem=0.2 python train1.py 3 > output_1_CV3
THEANO_FLAGS=device=gpu,floatX=float32,lib.cnmem=0.2 python train1.py 4 > output_1_CV4
