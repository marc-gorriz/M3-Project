#!/bin/bash

sbatch --gres gpu:5 -D ~/work/M3-Project/DL1 tgpu.sh

#echo /tmp/tgpu.sh_master05_${SLURM_JOB_ID}.out
