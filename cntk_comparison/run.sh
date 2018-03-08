#!/bin/sh

docker build . -t gradient_cntk
DIR=`pwd`
nvidia-docker run -it --rm gradient_cntk
