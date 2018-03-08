#!/bin/bash

source /cntk/activate-cntk

# Add path to missing libs
export LD_LIBRARY_PATH="$HOME/anaconda3/envs/cntk-py35/lib/python3.5/site-packages/cntk/libs:$LD_LIBRARY_PATH"
echo Library path $LD_LIBRARY_PATH

cntk
