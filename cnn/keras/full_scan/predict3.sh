#!/bin/bash

THEANO_FLAGS=device=gpu,floatX=float32 python predict.py 1 3
THEANO_FLAGS=device=gpu,floatX=float32 python predict.py 2 3
THEANO_FLAGS=device=gpu,floatX=float32 python predict.py 3 3
THEANO_FLAGS=device=gpu,floatX=float32 python predict.py 4 3
THEANO_FLAGS=device=gpu,floatX=float32 python predict.py 5 3
THEANO_FLAGS=device=gpu,floatX=float32 python predict.py 6 3
THEANO_FLAGS=device=gpu,floatX=float32 python predict.py 7 3
THEANO_FLAGS=device=gpu,floatX=float32 python predict.py 8 3
THEANO_FLAGS=device=gpu,floatX=float32 python predict.py 9 3
THEANO_FLAGS=device=gpu,floatX=float32 python predict.py 10 3
