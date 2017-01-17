#!/bin/bash

THEANO_FLAGS=device=gpu,floatX=float32 python train_GM.py 1 > output_GM_CV1
THEANO_FLAGS=device=gpu,floatX=float32 python train_GM.py 2 > output_GM_CV2
THEANO_FLAGS=device=gpu,floatX=float32 python train_GM.py 3 > output_GM_CV3
THEANO_FLAGS=device=gpu,floatX=float32 python train_GM.py 4 > output_GM_CV4
THEANO_FLAGS=device=gpu,floatX=float32 python train_GM.py 5 > output_GM_CV5
THEANO_FLAGS=device=gpu,floatX=float32 python train_GM.py 6 > output_GM_CV6
THEANO_FLAGS=device=gpu,floatX=float32 python train_GM.py 7 > output_GM_CV7
THEANO_FLAGS=device=gpu,floatX=float32 python train_GM.py 8 > output_GM_CV8
THEANO_FLAGS=device=gpu,floatX=float32 python train_GM.py 9 > output_GM_CV9
THEANO_FLAGS=device=gpu,floatX=float32 python train_GM.py 10 > output_GM_CV10
