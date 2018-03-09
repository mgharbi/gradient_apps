#!/bin/sh

docker build . -t gradient_cntk
DIR=`pwd`
nvidia-docker run -it --rm -v "$DIR/output":/output gradient_cntk
