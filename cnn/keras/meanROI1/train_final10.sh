#!/bin/bash

THEANO_FLAGS=device=gpu,floatX=float32 python train.py 1 10 > output_meanROI1_pretrained_final10_rui_CV1
THEANO_FLAGS=device=gpu,floatX=float32 python train.py 2 10 > output_meanROI1_pretrained_final10_rui_CV2
THEANO_FLAGS=device=gpu,floatX=float32 python train.py 3 10 > output_meanROI1_pretrained_final10_rui_CV3
THEANO_FLAGS=device=gpu,floatX=float32 python train.py 4 10 > output_meanROI1_pretrained_final10_rui_CV4
THEANO_FLAGS=device=gpu,floatX=float32 python train.py 5 10 > output_meanROI1_pretrained_final10_rui_CV5
THEANO_FLAGS=device=gpu,floatX=float32 python train.py 6 10 > output_meanROI1_pretrained_final10_rui_CV6
THEANO_FLAGS=device=gpu,floatX=float32 python train.py 7 10 > output_meanROI1_pretrained_final10_rui_CV7
THEANO_FLAGS=device=gpu,floatX=float32 python train.py 8 10 > output_meanROI1_pretrained_final10_rui_CV8
THEANO_FLAGS=device=gpu,floatX=float32 python train.py 9 10 > output_meanROI1_pretrained_final10_rui_CV9
THEANO_FLAGS=device=gpu,floatX=float32 python train.py 10 10 > output_meanROI1_pretrained_final10_rui_CV10

