#!/bin/bash

./predict1_3.sh &
./predict1_5.sh
./create_importanceMap.sh
./train2.sh

