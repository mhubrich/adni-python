#!/bin/bash

THEANO_FLAGS=device=gpu,floatX=float32,lib.cnmem=1.0 python -u train.py 1
THEANO_FLAGS=device=gpu,floatX=float32,lib.cnmem=1.0 python -u train.py 2
THEANO_FLAGS=device=gpu,floatX=float32,lib.cnmem=1.0 python -u train.py 3
THEANO_FLAGS=device=gpu,floatX=float32,lib.cnmem=1.0 python -u train.py 4
THEANO_FLAGS=device=gpu,floatX=float32,lib.cnmem=1.0 python -u train.py 5
THEANO_FLAGS=device=gpu,floatX=float32,lib.cnmem=1.0 python -u train.py 6
THEANO_FLAGS=device=gpu,floatX=float32,lib.cnmem=1.0 python -u train.py 7
THEANO_FLAGS=device=gpu,floatX=float32,lib.cnmem=1.0 python -u train.py 8
THEANO_FLAGS=device=gpu,floatX=float32,lib.cnmem=1.0 python -u train.py 9
THEANO_FLAGS=device=gpu,floatX=float32,lib.cnmem=1.0 python -u train.py 10

