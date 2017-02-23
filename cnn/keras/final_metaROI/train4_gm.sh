#!/bin/bash

THEANO_FLAGS=device=gpu,floatX=float32 python train4_gm.py 1 > output_4_gm_CV1
THEANO_FLAGS=device=gpu,floatX=float32 python train4_gm.py 2 > output_4_gm_CV2
THEANO_FLAGS=device=gpu,floatX=float32 python train4_gm.py 3 > output_4_gm_CV3
THEANO_FLAGS=device=gpu,floatX=float32 python train4_gm.py 4 > output_4_gm_CV4
THEANO_FLAGS=device=gpu,floatX=float32 python train4_gm.py 5 > output_4_gm_CV5
THEANO_FLAGS=device=gpu,floatX=float32 python train4_gm.py 6 > output_4_gm_CV6
THEANO_FLAGS=device=gpu,floatX=float32 python train4_gm.py 7 > output_4_gm_CV7
THEANO_FLAGS=device=gpu,floatX=float32 python train4_gm.py 8 > output_4_gm_CV8
THEANO_FLAGS=device=gpu,floatX=float32 python train4_gm.py 9 > output_4_gm_CV9
THEANO_FLAGS=device=gpu,floatX=float32 python train4_gm.py 10 > output_4_gm_CV10
