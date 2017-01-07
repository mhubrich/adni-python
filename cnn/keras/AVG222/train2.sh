#!/bin/bash

THEANO_FLAGS=device=gpu,floatX=float32,lib.cnmem=1.0 python -u train.py 5
THEANO_FLAGS=device=gpu,floatX=float32,lib.cnmem=1.0 python -u train.py 6
THEANO_FLAGS=device=gpu,floatX=float32,lib.cnmem=1.0 python -u train.py 7

