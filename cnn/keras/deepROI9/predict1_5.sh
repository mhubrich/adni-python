#!/bin/bash

THEANO_FLAGS=device=gpu,floatX=float32 python predict1.py 5 1
THEANO_FLAGS=device=gpu,floatX=float32 python predict1.py 5 2
THEANO_FLAGS=device=gpu,floatX=float32 python predict1.py 5 3
THEANO_FLAGS=device=gpu,floatX=float32 python predict1.py 5 4
THEANO_FLAGS=device=gpu,floatX=float32 python predict1.py 5 5
THEANO_FLAGS=device=gpu,floatX=float32 python predict1.py 5 6
THEANO_FLAGS=device=gpu,floatX=float32 python predict1.py 5 7
THEANO_FLAGS=device=gpu,floatX=float32 python predict1.py 5 8
THEANO_FLAGS=device=gpu,floatX=float32 python predict1.py 5 9
THEANO_FLAGS=device=gpu,floatX=float32 python predict1.py 5 10
