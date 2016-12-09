#!/bin/bash

THEANO_FLAGS=device=gpu,floatX=float32 python train_diff_single1.py 1 > output_meanROI1_pretrained_diff_CV1
THEANO_FLAGS=device=gpu,floatX=float32 python train_diff_single1.py 2 > output_meanROI1_pretrained_diff_CV2
THEANO_FLAGS=device=gpu,floatX=float32 python train_diff_single1.py 3 > output_meanROI1_pretrained_diff_CV3

