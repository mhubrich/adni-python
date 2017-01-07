#!/bin/bash

THEANO_FLAGS=device=gpu,floatX=float32 python train_AD.py 1 > output_deepROI2_AD_1_CV1
THEANO_FLAGS=device=gpu,floatX=float32 python train_AD.py 2 > output_deepROI2_AD_1_CV2
THEANO_FLAGS=device=gpu,floatX=float32 python train_AD.py 3 > output_deepROI2_AD_1_CV3
THEANO_FLAGS=device=gpu,floatX=float32 python train_AD.py 4 > output_deepROI2_AD_1_CV4
THEANO_FLAGS=device=gpu,floatX=float32 python train_AD.py 5 > output_deepROI2_AD_1_CV5

