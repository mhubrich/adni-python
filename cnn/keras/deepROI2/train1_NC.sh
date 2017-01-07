#!/bin/bash

THEANO_FLAGS=device=gpu,floatX=float32 python train_NC.py 1 > output_deepROI2_NC_1_CV1
THEANO_FLAGS=device=gpu,floatX=float32 python train_NC.py 2 > output_deepROI2_NC_1_CV2
THEANO_FLAGS=device=gpu,floatX=float32 python train_NC.py 3 > output_deepROI2_NC_1_CV3
THEANO_FLAGS=device=gpu,floatX=float32 python train_NC.py 4 > output_deepROI2_NC_1_CV4
THEANO_FLAGS=device=gpu,floatX=float32 python train_NC.py 5 > output_deepROI2_NC_1_CV5

