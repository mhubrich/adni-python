#!/bin/bash

THEANO_FLAGS=device=gpu,floatX=float32,lib.cnmem=1.0 python -u train.py 3

