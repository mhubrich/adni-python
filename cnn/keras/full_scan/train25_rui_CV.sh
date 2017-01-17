#!/bin/bash

pretrained=$1

THEANO_FLAGS=device=gpu,floatX=float32 python train25_rui_CV.py 1 $1 > output_25_rui_"${pretrained}"_CV1
THEANO_FLAGS=device=gpu,floatX=float32 python train25_rui_CV.py 2 $1 > output_25_rui_"${pretrained}"_CV2
THEANO_FLAGS=device=gpu,floatX=float32 python train25_rui_CV.py 3 $1 > output_25_rui_"${pretrained}"_CV3
THEANO_FLAGS=device=gpu,floatX=float32 python train25_rui_CV.py 4 $1 > output_25_rui_"${pretrained}"_CV4
THEANO_FLAGS=device=gpu,floatX=float32 python train25_rui_CV.py 5 $1 > output_25_rui_"${pretrained}"_CV5
THEANO_FLAGS=device=gpu,floatX=float32 python train25_rui_CV.py 6 $1 > output_25_rui_"${pretrained}"_CV6
THEANO_FLAGS=device=gpu,floatX=float32 python train25_rui_CV.py 7 $1 > output_25_rui_"${pretrained}"_CV7
THEANO_FLAGS=device=gpu,floatX=float32 python train25_rui_CV.py 8 $1 > output_25_rui_"${pretrained}"_CV8
THEANO_FLAGS=device=gpu,floatX=float32 python train25_rui_CV.py 9 $1 > output_25_rui_"${pretrained}"_CV9
THEANO_FLAGS=device=gpu,floatX=float32 python train25_rui_CV.py 10 $1 > output_25_rui_"${pretrained}"_CV10
