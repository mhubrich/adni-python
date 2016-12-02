#!/bin/bash

THEANO_FLAGS=device=gpu,floatX=float32 python train_rui.py 1 > output_norui_CV1
THEANO_FLAGS=device=gpu,floatX=float32 python train_rui.py 2 > output_norui_CV2
THEANO_FLAGS=device=gpu,floatX=float32 python train_rui.py 3 > output_norui_CV3
THEANO_FLAGS=device=gpu,floatX=float32 python train_rui.py 4 > output_norui_CV4
THEANO_FLAGS=device=gpu,floatX=float32 python train_rui.py 5 > output_norui_CV5
THEANO_FLAGS=device=gpu,floatX=float32 python train_rui.py 6 > output_norui_CV6
THEANO_FLAGS=device=gpu,floatX=float32 python train_rui.py 7 > output_norui_CV7
THEANO_FLAGS=device=gpu,floatX=float32 python train_rui.py 8 > output_norui_CV8
THEANO_FLAGS=device=gpu,floatX=float32 python train_rui.py 9 > output_norui_CV9
THEANO_FLAGS=device=gpu,floatX=float32 python train_rui.py 10 > output_norui_CV10
