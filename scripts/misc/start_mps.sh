#!/bin/bash

MPSDIR=/root/mps
rm -rf $MPSDIR

mkdir -p $MPSDIR
mkdir -p ${MPSDIR}/nvidia-mps
mkdir -p ${MPSDIR}/nvidia-log
chmod 777 ${MPSDIR}/nvidia-log

# export CUDA_VISIBLE_DEVICES=0
export CUDA_MPS_PIPE_DIRECTORY=${MPSDIR}/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=${MPSDIR}/nvidia-log

nvidia-cuda-mps-control -d

chmod 777 ${MPSDIR}/nvidia-log