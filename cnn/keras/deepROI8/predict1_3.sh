#!/bin/bash

THEANO_FLAGS=device=gpu,floatX=float32 python predict1.py 3 1
THEANO_FLAGS=device=gpu,floatX=float32 python predict1.py 3 2
THEANO_FLAGS=device=gpu,floatX=float32 python predict1.py 3 3
THEANO_FLAGS=device=gpu,floatX=float32 python predict1.py 3 4
THEANO_FLAGS=device=gpu,floatX=float32 python predict1.py 3 5
THEANO_FLAGS=device=gpu,floatX=float32 python predict1.py 3 6
THEANO_FLAGS=device=gpu,floatX=float32 python predict1.py 3 7
THEANO_FLAGS=device=gpu,floatX=float32 python predict1.py 3 8
THEANO_FLAGS=device=gpu,floatX=float32 python predict1.py 3 9
THEANO_FLAGS=device=gpu,floatX=float32 python predict1.py 3 10
