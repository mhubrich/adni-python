#!/bin/bash

THEANO_FLAGS=device=gpu,floatX=float32 python train5_2_pSMC.py 1 > output_5_2_pSMC_CV1
THEANO_FLAGS=device=gpu,floatX=float32 python train5_2_pSMC.py 2 > output_5_2_pSMC_CV2
THEANO_FLAGS=device=gpu,floatX=float32 python train5_2_pSMC.py 3 > output_5_2_pSMC_CV3
THEANO_FLAGS=device=gpu,floatX=float32 python train5_2_pSMC.py 4 > output_5_2_pSMC_CV4
THEANO_FLAGS=device=gpu,floatX=float32 python train5_2_pSMC.py 5 > output_5_2_pSMC_CV5
THEANO_FLAGS=device=gpu,floatX=float32 python train5_2_pSMC.py 6 > output_5_2_pSMC_CV6
THEANO_FLAGS=device=gpu,floatX=float32 python train5_2_pSMC.py 7 > output_5_2_pSMC_CV7
THEANO_FLAGS=device=gpu,floatX=float32 python train5_2_pSMC.py 8 > output_5_2_pSMC_CV8
THEANO_FLAGS=device=gpu,floatX=float32 python train5_2_pSMC.py 9 > output_5_2_pSMC_CV9
THEANO_FLAGS=device=gpu,floatX=float32 python train5_2_pSMC.py 10 > output_5_2_pSMC_CV10
