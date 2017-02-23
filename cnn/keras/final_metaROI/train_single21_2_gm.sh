#!/bin/bash

THEANO_FLAGS=device=gpu,floatX=float32 python train_single21_2_gm.py 3 > output_single21_2_gm_CV3
THEANO_FLAGS=device=gpu,floatX=float32 python train_single21_2_gm.py 4 > output_single21_2_gm_CV4
THEANO_FLAGS=device=gpu,floatX=float32 python train_single22_2_gm.py 3 > output_single22_2_gm_CV3
THEANO_FLAGS=device=gpu,floatX=float32 python train_single23_2_gm.py 4 > output_single23_2_gm_CV4
THEANO_FLAGS=device=gpu,floatX=float32 python train_single24_2_gm.py 3 > output_single24_2_gm_CV3
THEANO_FLAGS=device=gpu,floatX=float32 python train_single24_2_gm.py 4 > output_single24_2_gm_CV4
THEANO_FLAGS=device=gpu,floatX=float32 python train_single25_2_gm.py 3 > output_single25_2_gm_CV3
