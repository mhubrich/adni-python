#!/bin/bash

THEANO_FLAGS=device=gpu,floatX=float32 python train.py 1 8 > output_meanROI1_pretrained_final8_rui_CV1
THEANO_FLAGS=device=gpu,floatX=float32 python train.py 2 8 > output_meanROI1_pretrained_final8_rui_CV2
THEANO_FLAGS=device=gpu,floatX=float32 python train.py 3 8 > output_meanROI1_pretrained_final8_rui_CV3
THEANO_FLAGS=device=gpu,floatX=float32 python train.py 4 8 > output_meanROI1_pretrained_final8_rui_CV4
THEANO_FLAGS=device=gpu,floatX=float32 python train.py 5 8 > output_meanROI1_pretrained_final8_rui_CV5
THEANO_FLAGS=device=gpu,floatX=float32 python train.py 6 8 > output_meanROI1_pretrained_final8_rui_CV6
THEANO_FLAGS=device=gpu,floatX=float32 python train.py 7 8 > output_meanROI1_pretrained_final8_rui_CV7
THEANO_FLAGS=device=gpu,floatX=float32 python train.py 8 8 > output_meanROI1_pretrained_final8_rui_CV8
THEANO_FLAGS=device=gpu,floatX=float32 python train.py 9 8 > output_meanROI1_pretrained_final8_rui_CV9
THEANO_FLAGS=device=gpu,floatX=float32 python train.py 10 8 > output_meanROI1_pretrained_final8_rui_CV10

