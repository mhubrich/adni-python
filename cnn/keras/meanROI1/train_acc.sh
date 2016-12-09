#!/bin/bash

THEANO_FLAGS=device=gpu,floatX=float32 python train_acc.py 1 > output_meanROI1_pretrained_acc_CV1
THEANO_FLAGS=device=gpu,floatX=float32 python train_acc.py 2 > output_meanROI1_pretrained_acc_CV2
THEANO_FLAGS=device=gpu,floatX=float32 python train_acc.py 3 > output_meanROI1_pretrained_acc_CV3
THEANO_FLAGS=device=gpu,floatX=float32 python train_acc.py 4 > output_meanROI1_pretrained_acc_CV4
THEANO_FLAGS=device=gpu,floatX=float32 python train_acc.py 5 > output_meanROI1_pretrained_acc_CV5
THEANO_FLAGS=device=gpu,floatX=float32 python train_acc.py 6 > output_meanROI1_pretrained_acc_CV6
THEANO_FLAGS=device=gpu,floatX=float32 python train_acc.py 7 > output_meanROI1_pretrained_acc_CV7
THEANO_FLAGS=device=gpu,floatX=float32 python train_acc.py 8 > output_meanROI1_pretrained_acc_CV8
THEANO_FLAGS=device=gpu,floatX=float32 python train_acc.py 9 > output_meanROI1_pretrained_acc_CV9
THEANO_FLAGS=device=gpu,floatX=float32 python train_acc.py 10 > output_meanROI1_pretrained_acc_CV10

