#!/bin/bash

THEANO_FLAGS=device=gpu,floatX=float32,lib.cnmem=1.0 python -u train.py 8
THEANO_FLAGS=device=gpu,floatX=float32,lib.cnmem=1.0 python -u train.py 9
THEANO_FLAGS=device=gpu,floatX=float32,lib.cnmem=1.0 python -u train.py 10

