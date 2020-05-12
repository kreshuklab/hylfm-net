#!/bin/bash
#SBATCH -N 1                         # number of nodes
#SBATCH -n 8                         # number of cores
#SBATCH --mem 36GB                   # memory pool for all cores
#SBATCH -t 6-00:01:00                # runtime limit (D-HH:MM:SS)
#SBATCH -o slurm_out/%N.%j.out       # STDOUT
#SBATCH -e slurm_out/%N.%j.err      # STDERR
#SBATCH --mail-type=END,FAIL         # notifications for job done & fail
#SBATCH --mail-user=beuttenm@embl.de # send-to address
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -Cgpu=2080Ti
#SBATCH -J lnet

/g/kreshuk/beuttenm/miniconda3/envs/llnet/bin/python -m lnet $1
