#!/bin/bash

THEANO_FLAGS=device=gpu,floatX=float32 python train.py 1 > output_deepROI2_merged_2_CV1
THEANO_FLAGS=device=gpu,floatX=float32 python train.py 2 > output_deepROI2_merged_2_CV2
THEANO_FLAGS=device=gpu,floatX=float32 python train.py 3 > output_deepROI2_merged_2_CV3
THEANO_FLAGS=device=gpu,floatX=float32 python train.py 4 > output_deepROI2_merged_2_CV4
THEANO_FLAGS=device=gpu,floatX=float32 python train.py 5 > output_deepROI2_merged_2_CV5

