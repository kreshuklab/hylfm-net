#!/bin/bash
for i in {0..17}
do
  echo "submit $1 $i"
 sbatch prep.sh $1 $i
done

sleep 2
squeue -u beuttenm
