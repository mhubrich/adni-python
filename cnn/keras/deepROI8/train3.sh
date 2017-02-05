#!/bin/bash

THEANO_FLAGS=device=gpu,floatX=float32,nocleanup=True python train3.py 1 > output_3_CV1
