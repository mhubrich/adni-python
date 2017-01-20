#!/bin/bash

THEANO_FLAGS=device=gpu,floatX=float32 python train_baseline.py 1 > output_baseline_CV1
THEANO_FLAGS=device=gpu,floatX=float32 python train_baseline.py 2 > output_baseline_CV2
THEANO_FLAGS=device=gpu,floatX=float32 python train_baseline.py 3 > output_baseline_CV3
THEANO_FLAGS=device=gpu,floatX=float32 python train_baseline.py 4 > output_baseline_CV4
THEANO_FLAGS=device=gpu,floatX=float32 python train_baseline.py 5 > output_baseline_CV5
THEANO_FLAGS=device=gpu,floatX=float32 python train_baseline.py 6 > output_baseline_CV6
THEANO_FLAGS=device=gpu,floatX=float32 python train_baseline.py 7 > output_baseline_CV7
THEANO_FLAGS=device=gpu,floatX=float32 python train_baseline.py 8 > output_baseline_CV8
THEANO_FLAGS=device=gpu,floatX=float32 python train_baseline.py 9 > output_baseline_CV9
THEANO_FLAGS=device=gpu,floatX=float32 python train_baseline.py 10 > output_baseline_CV10
