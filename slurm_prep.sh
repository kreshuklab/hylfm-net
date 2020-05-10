#!/bin/bash
for i in {1..$2}
do
  echo "submit $1 $i"
#  sbatch prep.sh $1 $i
done
