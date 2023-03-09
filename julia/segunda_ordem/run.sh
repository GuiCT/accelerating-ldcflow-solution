#!/bin/bash
export JULIA_NUM_THREADS=$1
export OPENBLAS_NUM_THREADS=$2
julia -i ./cavidade.jl
exit
