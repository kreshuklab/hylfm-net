#!/bin/bash
#SBATCH -N 1                         # number of nodes
#SBATCH -n 8                         # number of cores
#SBATCH --mem 24GB                   # memory pool for all cores
#SBATCH -t 6-23:01:00                # runtime limit (D-HH:MM:SS)
#SBATCH -o /scratch/beuttenm/hylfm/slurm_out/%N.%j.out       # STDOUT
#SBATCH -e /scratch/beuttenm/hylfm/slurm_out/%N.%j.err       # STDERR
#SBATCH --mail-type=END,FAIL         # notifications for job done & fail
#SBATCH --mail-user=beuttenm@embl.de # send-to address
#SBATCH -J hylfm-cpu
#SBATCH --nice=10

/g/kreshuk/beuttenm/miniconda3/envs/hylfm/bin/python -Om hylfm "$@"
