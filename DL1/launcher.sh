#!/bin/bash

sbatch --gres gpu:1 -D ~/work/M3-Project/DL1/"$1" tgpu.sh "$1"
