#!/bin/bash

THEANO_FLAGS=device=gpu,floatX=float32 python train1_gm_norui.py 1 > output_gm_norui_1_CV1
THEANO_FLAGS=device=gpu,floatX=float32 python train1_gm_norui.py 2 > output_gm_norui_1_CV2
THEANO_FLAGS=device=gpu,floatX=float32 python train1_gm_norui.py 3 > output_gm_norui_1_CV3
THEANO_FLAGS=device=gpu,floatX=float32 python train1_gm_norui.py 4 > output_gm_norui_1_CV4
THEANO_FLAGS=device=gpu,floatX=float32 python train1_gm_norui.py 5 > output_gm_norui_1_CV5
THEANO_FLAGS=device=gpu,floatX=float32 python train1_gm_norui.py 6 > output_gm_norui_1_CV6
THEANO_FLAGS=device=gpu,floatX=float32 python train1_gm_norui.py 7 > output_gm_norui_1_CV7
THEANO_FLAGS=device=gpu,floatX=float32 python train1_gm_norui.py 8 > output_gm_norui_1_CV8
THEANO_FLAGS=device=gpu,floatX=float32 python train1_gm_norui.py 9 > output_gm_norui_1_CV9
THEANO_FLAGS=device=gpu,floatX=float32 python train1_gm_norui.py 10 > output_gm_norui_1_CV10
