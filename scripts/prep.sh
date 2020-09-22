#!/bin/bash
#SBATCH -N 1                         # number of nodes
#SBATCH -n 8                         # number of cores
#SBATCH --mem 20GB                   # memory pool for all cores
#SBATCH -t 0-10:01:00                # runtime limit (D-HH:MM:SS)
#SBATCH -o /scratch/beuttenm/lnet/slurm_out_prep/%N_%A_%a.out  # STDOUT
#SBATCH -e /scratch/beuttenm/lnet/slurm_out_prep/%N_%A_%a.err  # STDERR
#SBATCH --mail-type=END,FAIL,ARRAY_TASKS                          # notifications for job done & fail per task
#SBATCH --mail-user=beuttenm@embl.de                              # send-to address
#SBATCH -J lnet_prep
#SBATCH --array=0-0%1

/g/kreshuk/beuttenm/miniconda3/envs/llnet/bin/python $1 $SLURM_ARRAY_TASK_ID $2
