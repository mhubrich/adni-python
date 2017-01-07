#!/bin/bash

THEANO_FLAGS=device=gpu,floatX=float32 python train.py 1 5 > output_meanROI1_pretrained_final5_rui_CV1
THEANO_FLAGS=device=gpu,floatX=float32 python train.py 2 5 > output_meanROI1_pretrained_final5_rui_CV2
THEANO_FLAGS=device=gpu,floatX=float32 python train.py 3 5 > output_meanROI1_pretrained_final5_rui_CV3
THEANO_FLAGS=device=gpu,floatX=float32 python train.py 4 5 > output_meanROI1_pretrained_final5_rui_CV4
THEANO_FLAGS=device=gpu,floatX=float32 python train.py 5 5 > output_meanROI1_pretrained_final5_rui_CV5
THEANO_FLAGS=device=gpu,floatX=float32 python train.py 6 5 > output_meanROI1_pretrained_final5_rui_CV6
THEANO_FLAGS=device=gpu,floatX=float32 python train.py 7 5 > output_meanROI1_pretrained_final5_rui_CV7
THEANO_FLAGS=device=gpu,floatX=float32 python train.py 8 5 > output_meanROI1_pretrained_final5_rui_CV8
THEANO_FLAGS=device=gpu,floatX=float32 python train.py 9 5 > output_meanROI1_pretrained_final5_rui_CV9
THEANO_FLAGS=device=gpu,floatX=float32 python train.py 10 5 > output_meanROI1_pretrained_final5_rui_CV10

