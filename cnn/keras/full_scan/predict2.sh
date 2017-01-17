#!/bin/bash

THEANO_FLAGS=device=gpu,floatX=float32 python predict.py 1 2
THEANO_FLAGS=device=gpu,floatX=float32 python predict.py 2 2
THEANO_FLAGS=device=gpu,floatX=float32 python predict.py 3 2
THEANO_FLAGS=device=gpu,floatX=float32 python predict.py 4 2
THEANO_FLAGS=device=gpu,floatX=float32 python predict.py 5 2
THEANO_FLAGS=device=gpu,floatX=float32 python predict.py 6 2
THEANO_FLAGS=device=gpu,floatX=float32 python predict.py 7 2
THEANO_FLAGS=device=gpu,floatX=float32 python predict.py 8 2
THEANO_FLAGS=device=gpu,floatX=float32 python predict.py 9 2
THEANO_FLAGS=device=gpu,floatX=float32 python predict.py 10 2
