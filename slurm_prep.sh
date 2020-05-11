#!/bin/bash
for i in {0..17}
do
  echo "submit $1 $i"  
  sbatch prep.sh $1 $i
done

sleep 10
squeue -u beuttenm

# for i in {0..16}
# do
#   echo "submit $1 $i"  
#   ./prep.sh $1 $i &
# #  sbatch prep.sh $1 $i
#  sbatch prep.sh $1 $i
# done

# # wait
# sleep 2
# squeue -u beuttenm
