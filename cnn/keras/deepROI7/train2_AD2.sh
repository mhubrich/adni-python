#!/bin/bash

THEANO_FLAGS=device=gpu,floatX=float32,lib.cnmem=1.0 python train_AE_AD_2.py 4 > output_AE_AD_2_CV4
THEANO_FLAGS=device=gpu,floatX=float32,lib.cnmem=1.0 python train_AE_AD_2.py 5 > output_AE_AD_2_CV5
THEANO_FLAGS=device=gpu,floatX=float32,lib.cnmem=1.0 python train_AE_AD_2.py 6 > output_AE_AD_2_CV6

