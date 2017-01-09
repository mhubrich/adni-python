#!/bin/bash

THEANO_FLAGS=device=gpu,floatX=float32 python train3.py 1 > output_3_CV1
THEANO_FLAGS=device=gpu,floatX=float32 python train3.py 2 > output_3_CV2
THEANO_FLAGS=device=gpu,floatX=float32 python train3.py 3 > output_3_CV3
THEANO_FLAGS=device=gpu,floatX=float32 python train3.py 4 > output_3_CV4
THEANO_FLAGS=device=gpu,floatX=float32 python train3.py 5 > output_3_CV5
