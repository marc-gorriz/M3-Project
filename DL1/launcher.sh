#!/bin/bash

sbatch --gres gpu:1 --time=72:00:00 -D ~/work/M3-Project/DL1 tgpu.sh "$1"

#echo /tmp/tgpu.sh_master05_${SLURM_JOB_ID}.out
