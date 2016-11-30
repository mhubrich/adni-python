#!/bin/bash

THEANO_FLAGS=device=gpu,floatX=float32 python train_single_diff.py 1 > output_meanROI1_diff_CV1
THEANO_FLAGS=device=gpu,floatX=float32 python train_single_diff.py 2 > output_meanROI1_diff_CV2
THEANO_FLAGS=device=gpu,floatX=float32 python train_single_diff.py 3 > output_meanROI1_diff_CV3
THEANO_FLAGS=device=gpu,floatX=float32 python train_single_diff.py 4 > output_meanROI1_diff_CV4
THEANO_FLAGS=device=gpu,floatX=float32 python train_single_diff.py 5 > output_meanROI1_diff_CV5
THEANO_FLAGS=device=gpu,floatX=float32 python train_single_diff.py 6 > output_meanROI1_diff_CV6
THEANO_FLAGS=device=gpu,floatX=float32 python train_single_diff.py 7 > output_meanROI1_diff_CV7
THEANO_FLAGS=device=gpu,floatX=float32 python train_single_diff.py 8 > output_meanROI1_diff_CV8
THEANO_FLAGS=device=gpu,floatX=float32 python train_single_diff.py 9 > output_meanROI1_diff_CV9
THEANO_FLAGS=device=gpu,floatX=float32 python train_single_diff.py 10 > output_meanROI1_diff_CV10

