#!/bin/bash

THEANO_FLAGS=device=gpu,floatX=float32,lib.cnmem=1.0 python -u train.py 1
THEANO_FLAGS=device=gpu,floatX=float32,lib.cnmem=1.0 python -u train.py 2
#THEANO_FLAGS=device=gpu,floatX=float32,lib.cnmem=1.0 python -u train.py 3

