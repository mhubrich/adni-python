#!/bin/bash

THEANO_FLAGS=device=gpu,floatX=float32 python train_single5.py 1 > output_meanROI1_pretrained5_rui_CV1
THEANO_FLAGS=device=gpu,floatX=float32 python train_single5.py 2 > output_meanROI1_pretrained5_rui_CV2
THEANO_FLAGS=device=gpu,floatX=float32 python train_single5.py 3 > output_meanROI1_pretrained5_rui_CV3
THEANO_FLAGS=device=gpu,floatX=float32 python train_single5.py 4 > output_meanROI1_pretrained5_rui_CV4
THEANO_FLAGS=device=gpu,floatX=float32 python train_single5.py 5 > output_meanROI1_pretrained5_rui_CV5
THEANO_FLAGS=device=gpu,floatX=float32 python train_single5.py 6 > output_meanROI1_pretrained5_rui_CV6
THEANO_FLAGS=device=gpu,floatX=float32 python train_single5.py 7 > output_meanROI1_pretrained5_rui_CV7
THEANO_FLAGS=device=gpu,floatX=float32 python train_single5.py 8 > output_meanROI1_pretrained5_rui_CV8
THEANO_FLAGS=device=gpu,floatX=float32 python train_single5.py 9 > output_meanROI1_pretrained5_rui_CV9
THEANO_FLAGS=device=gpu,floatX=float32 python train_single5.py 10 > output_meanROI1_pretrained5_rui_CV10

