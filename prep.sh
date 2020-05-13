#!/bin/bash
#SBATCH -N 1                         # number of nodes
#SBATCH -n 16                        # number of cores
#SBATCH --mem 32GB                   # memory pool for all cores
#SBATCH -t 0-05:01:00                # runtime limit (D-HH:MM:SS)
#SBATCH -o slurm_out/%N.%j.out       # STDOUT
#SBATCH -e slurm_out/%N.%j.err       # STDERR
#SBATCH --mail-type=END,FAIL         # notifications for job done & fail
#SBATCH --mail-user=beuttenm@embl.de # send-to address
#SBATCH -J heart_dynamic_prep

/g/kreshuk/beuttenm/miniconda3/envs/llnet/bin/python $@
