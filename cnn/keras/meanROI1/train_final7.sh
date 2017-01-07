#!/bin/bash

THEANO_FLAGS=device=gpu,floatX=float32 python train.py 1 7 > output_meanROI1_pretrained_final7_rui_CV1
THEANO_FLAGS=device=gpu,floatX=float32 python train.py 2 7 > output_meanROI1_pretrained_final7_rui_CV2
THEANO_FLAGS=device=gpu,floatX=float32 python train.py 3 7 > output_meanROI1_pretrained_final7_rui_CV3
THEANO_FLAGS=device=gpu,floatX=float32 python train.py 4 7 > output_meanROI1_pretrained_final7_rui_CV4
THEANO_FLAGS=device=gpu,floatX=float32 python train.py 5 7 > output_meanROI1_pretrained_final7_rui_CV5
THEANO_FLAGS=device=gpu,floatX=float32 python train.py 6 7 > output_meanROI1_pretrained_final7_rui_CV6
THEANO_FLAGS=device=gpu,floatX=float32 python train.py 7 7 > output_meanROI1_pretrained_final7_rui_CV7
THEANO_FLAGS=device=gpu,floatX=float32 python train.py 8 7 > output_meanROI1_pretrained_final7_rui_CV8
THEANO_FLAGS=device=gpu,floatX=float32 python train.py 9 7 > output_meanROI1_pretrained_final7_rui_CV9
THEANO_FLAGS=device=gpu,floatX=float32 python train.py 10 7 > output_meanROI1_pretrained_final7_rui_CV10

