#!/bin/bash
#SBATCH -A m4335
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 1:00:00
#SBATCH -N 10
#SBATCH --ntasks-per-node=4
#SBATCH -c 32
#SBATCH --gpu-bind=none

export nnodes=25 #${SLURM_JOB_NUM_NODES}

srun --ntasks-per-node=4 --gpus-per-node=4 -c 32 -N ${nnodes} python test_snap.py 1 7 ${nnodes}