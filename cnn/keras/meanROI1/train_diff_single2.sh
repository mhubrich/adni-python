#!/bin/bash

THEANO_FLAGS=device=gpu,floatX=float32 python train_diff_single1.py 4 > output_meanROI1_pretrained_diff_CV4
THEANO_FLAGS=device=gpu,floatX=float32 python train_diff_single1.py 5 > output_meanROI1_pretrained_diff_CV5
THEANO_FLAGS=device=gpu,floatX=float32 python train_diff_single1.py 6 > output_meanROI1_pretrained_diff_CV6

