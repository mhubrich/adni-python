#!/bin/bash
# $1: filter length
# $2: fold of trained CNN

THEANO_FLAGS=device=gpu,floatX=float32 python predict1_full_all.py 1 1 &
THEANO_FLAGS=device=gpu,floatX=float32 python predict1_full_all.py 3 1 &
THEANO_FLAGS=device=gpu,floatX=float32 python predict1_full_all.py 5 1
THEANO_FLAGS=device=gpu,floatX=float32 python predict1_full_all.py 1 2 &
THEANO_FLAGS=device=gpu,floatX=float32 python predict1_full_all.py 3 2 &
THEANO_FLAGS=device=gpu,floatX=float32 python predict1_full_all.py 5 2
THEANO_FLAGS=device=gpu,floatX=float32 python predict1_full_all.py 1 3 &
THEANO_FLAGS=device=gpu,floatX=float32 python predict1_full_all.py 3 3 &
THEANO_FLAGS=device=gpu,floatX=float32 python predict1_full_all.py 5 3
THEANO_FLAGS=device=gpu,floatX=float32 python predict1_full_all.py 1 4 &
THEANO_FLAGS=device=gpu,floatX=float32 python predict1_full_all.py 3 4 &
THEANO_FLAGS=device=gpu,floatX=float32 python predict1_full_all.py 5 4
THEANO_FLAGS=device=gpu,floatX=float32 python predict1_full_all.py 1 5 &
THEANO_FLAGS=device=gpu,floatX=float32 python predict1_full_all.py 3 5 &
THEANO_FLAGS=device=gpu,floatX=float32 python predict1_full_all.py 5 5
THEANO_FLAGS=device=gpu,floatX=float32 python predict1_full_all.py 1 6 &
THEANO_FLAGS=device=gpu,floatX=float32 python predict1_full_all.py 3 6 &
THEANO_FLAGS=device=gpu,floatX=float32 python predict1_full_all.py 5 6
THEANO_FLAGS=device=gpu,floatX=float32 python predict1_full_all.py 1 7 &
THEANO_FLAGS=device=gpu,floatX=float32 python predict1_full_all.py 3 7 &
THEANO_FLAGS=device=gpu,floatX=float32 python predict1_full_all.py 5 7
THEANO_FLAGS=device=gpu,floatX=float32 python predict1_full_all.py 1 8 &
THEANO_FLAGS=device=gpu,floatX=float32 python predict1_full_all.py 3 8 &
THEANO_FLAGS=device=gpu,floatX=float32 python predict1_full_all.py 5 8
THEANO_FLAGS=device=gpu,floatX=float32 python predict1_full_all.py 1 9 &
THEANO_FLAGS=device=gpu,floatX=float32 python predict1_full_all.py 3 9 &
THEANO_FLAGS=device=gpu,floatX=float32 python predict1_full_all.py 5 9
THEANO_FLAGS=device=gpu,floatX=float32 python predict1_full_all.py 1 10 &
THEANO_FLAGS=device=gpu,floatX=float32 python predict1_full_all.py 3 10 &
THEANO_FLAGS=device=gpu,floatX=float32 python predict1_full_all.py 5 10
