#!/bin/bash

mkdir ~/work/DL1-OUTPUT/"$1"
sbatch --gres gpu:1 -D ~/work/DL1-OUTPUT/"$1" tgpu.sh "$1"
