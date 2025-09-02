#!/bin/bash
# the following must be performed with root privilege
# >>> sudo sh scripts/start_mps.sh [MPS_DIR]

BASEDIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
MPSDIR=${BASEDIR}/log/mps

export CUDA_MPS_PIPE_DIRECTORY=${MPSDIR}/nvidia-mps
echo quit | nvidia-cuda-mps-control