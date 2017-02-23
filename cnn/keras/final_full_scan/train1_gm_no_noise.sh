#!/bin/bash

THEANO_FLAGS=device=gpu,floatX=float32 python train1_gm_no_noise.py 1 > output_no_noise_1_CV1
THEANO_FLAGS=device=gpu,floatX=float32 python train1_gm_no_noise.py 2 > output_no_noise_1_CV2
THEANO_FLAGS=device=gpu,floatX=float32 python train1_gm_no_noise.py 3 > output_no_noise_1_CV3
THEANO_FLAGS=device=gpu,floatX=float32 python train1_gm_no_noise.py 4 > output_no_noise_1_CV4
THEANO_FLAGS=device=gpu,floatX=float32 python train1_gm_no_noise.py 5 > output_no_noise_1_CV5
THEANO_FLAGS=device=gpu,floatX=float32 python train1_gm_no_noise.py 6 > output_no_noise_1_CV6
THEANO_FLAGS=device=gpu,floatX=float32 python train1_gm_no_noise.py 7 > output_no_noise_1_CV7
THEANO_FLAGS=device=gpu,floatX=float32 python train1_gm_no_noise.py 8 > output_no_noise_1_CV8
THEANO_FLAGS=device=gpu,floatX=float32 python train1_gm_no_noise.py 9 > output_no_noise_1_CV9
THEANO_FLAGS=device=gpu,floatX=float32 python train1_gm_no_noise.py 10 > output_no_noise_1_CV10
