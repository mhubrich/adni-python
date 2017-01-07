#!/bin/bash

THEANO_FLAGS=device=gpu,floatX=float32 python train_AD.py 6 > output_deepROI2_AD_1_CV6
THEANO_FLAGS=device=gpu,floatX=float32 python train_AD.py 7 > output_deepROI2_AD_1_CV7
THEANO_FLAGS=device=gpu,floatX=float32 python train_AD.py 8 > output_deepROI2_AD_1_CV8
THEANO_FLAGS=device=gpu,floatX=float32 python train_AD.py 9 > output_deepROI2_AD_1_CV9
THEANO_FLAGS=device=gpu,floatX=float32 python train_AD.py 10 > output_deepROI2_AD_1_CV10

