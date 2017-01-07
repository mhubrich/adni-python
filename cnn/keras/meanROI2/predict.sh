#!/bin/bash

THEANO_FLAGS=device=gpu,floatX=float32 python predict.py 1 > output_predicted_filter_1_CV1
THEANO_FLAGS=device=gpu,floatX=float32 python predict.py 2 > output_predicted_filter_1_CV2
THEANO_FLAGS=device=gpu,floatX=float32 python predict.py 3 > output_predicted_filter_1_CV3
THEANO_FLAGS=device=gpu,floatX=float32 python predict.py 4 > output_predicted_filter_1_CV4
THEANO_FLAGS=device=gpu,floatX=float32 python predict.py 5 > output_predicted_filter_1_CV5
THEANO_FLAGS=device=gpu,floatX=float32 python predict.py 6 > output_predicted_filter_1_CV6
THEANO_FLAGS=device=gpu,floatX=float32 python predict.py 7 > output_predicted_filter_1_CV7
THEANO_FLAGS=device=gpu,floatX=float32 python predict.py 8 > output_predicted_filter_1_CV8
THEANO_FLAGS=device=gpu,floatX=float32 python predict.py 9 > output_predicted_filter_1_CV9
THEANO_FLAGS=device=gpu,floatX=float32 python predict.py 10 > output_predicted_filter_1_CV10

