#!/bin/bash

#THEANO_FLAGS=device=gpu,floatX=float32 python train.py 1 > output_meanROI1_pretrained_82_2_rui_finalCV1
THEANO_FLAGS=device=gpu,floatX=float32 python train.py 2 > output_meanROI1_pretrained_82_2_rui_finalCV2
THEANO_FLAGS=device=gpu,floatX=float32 python train.py 3 > output_meanROI1_pretrained_82_2_rui_finalCV3
THEANO_FLAGS=device=gpu,floatX=float32 python train.py 4 > output_meanROI1_pretrained_82_2_rui_finalCV4
THEANO_FLAGS=device=gpu,floatX=float32 python train.py 5 > output_meanROI1_pretrained_82_2_rui_finalCV5
THEANO_FLAGS=device=gpu,floatX=float32 python train.py 6 > output_meanROI1_pretrained_82_2_rui_finalCV6
THEANO_FLAGS=device=gpu,floatX=float32 python train.py 7 > output_meanROI1_pretrained_82_2_rui_finalCV7
THEANO_FLAGS=device=gpu,floatX=float32 python train.py 8 > output_meanROI1_pretrained_82_2_rui_finalCV8
#THEANO_FLAGS=device=gpu,floatX=float32 python train.py 9 > output_meanROI1_pretrained_82_2_rui_finalCV9
#THEANO_FLAGS=device=gpu,floatX=float32 python train.py 10 > output_meanROI1_pretrained_82_2_rui_finalCV10

