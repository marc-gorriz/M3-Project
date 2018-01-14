#!/bin/bash
#SBATCH -n 4                    # Number of cores
#SBATCH -N 1                    # Ensure that all cores are on one machine
#SBATCH -D /tmp                 # working directory
#SBATCH -t 0-00:05              # Runtime in D-HH:MM
#SBATCH -p dcc                  # Partition to submit to
#SBATCH --gres gpu:1            # Number of gpus required
#SBATCH -o %x_%u_%j.out         # File to which STDOUT will be written
#SBATCH -e %x_%u_%j.err         # File to which STDERR will be written

python ~/work/M3-Project/DL1/mlp_MIT_8_scene"$1".py "$1"
