#!/bin/bash

THEANO_FLAGS=device=gpu,floatX=float32 python train1_pSMC.py 1 > output_pSMC_1_CV1
THEANO_FLAGS=device=gpu,floatX=float32 python train1_pSMC.py 2 > output_pSMC_1_CV2
THEANO_FLAGS=device=gpu,floatX=float32 python train1_pSMC.py 3 > output_pSMC_1_CV3
THEANO_FLAGS=device=gpu,floatX=float32 python train1_pSMC.py 4 > output_pSMC_1_CV4
THEANO_FLAGS=device=gpu,floatX=float32 python train1_pSMC.py 5 > output_pSMC_1_CV5
THEANO_FLAGS=device=gpu,floatX=float32 python train1_pSMC.py 6 > output_pSMC_1_CV6
THEANO_FLAGS=device=gpu,floatX=float32 python train1_pSMC.py 7 > output_pSMC_1_CV7
THEANO_FLAGS=device=gpu,floatX=float32 python train1_pSMC.py 8 > output_pSMC_1_CV8
THEANO_FLAGS=device=gpu,floatX=float32 python train1_pSMC.py 9 > output_pSMC_1_CV9
THEANO_FLAGS=device=gpu,floatX=float32 python train1_pSMC.py 10 > output_pSMC_1_CV10
