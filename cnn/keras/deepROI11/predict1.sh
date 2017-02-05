#!/bin/bash
# $1: filter length
# $2: fold of trained CNN

THEANO_FLAGS=device=gpu,floatX=float32 python predict1.py $1 1 $2
THEANO_FLAGS=device=gpu,floatX=float32 python predict1.py $1 2 $2
THEANO_FLAGS=device=gpu,floatX=float32 python predict1.py $1 3 $2
THEANO_FLAGS=device=gpu,floatX=float32 python predict1.py $1 4 $2
THEANO_FLAGS=device=gpu,floatX=float32 python predict1.py $1 5 $2
THEANO_FLAGS=device=gpu,floatX=float32 python predict1.py $1 6 $2
THEANO_FLAGS=device=gpu,floatX=float32 python predict1.py $1 7 $2
THEANO_FLAGS=device=gpu,floatX=float32 python predict1.py $1 8 $2
THEANO_FLAGS=device=gpu,floatX=float32 python predict1.py $1 9 $2
THEANO_FLAGS=device=gpu,floatX=float32 python predict1.py $1 10 $2
