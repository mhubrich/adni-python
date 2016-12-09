#!/bin/bash

THEANO_FLAGS=device=gpu,floatX=float32 python train_diff_single1.py 7 > output_meanROI1_pretrained_diff_CV7
THEANO_FLAGS=device=gpu,floatX=float32 python train_diff_single1.py 8 > output_meanROI1_pretrained_diff_CV8
THEANO_FLAGS=device=gpu,floatX=float32 python train_diff_single1.py 9 > output_meanROI1_pretrained_diff_CV9
THEANO_FLAGS=device=gpu,floatX=float32 python train_diff_single1.py 10 > output_meanROI1_pretrained_diff_CV10

