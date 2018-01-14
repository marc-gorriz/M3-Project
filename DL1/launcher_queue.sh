#!/bin/bash

FLAG=true
#JOBID=0
   
#for i in `seq 1 2`; do
for i in {15..1}; do
  
   while $FLAG; do
      GPU=$(squeue | grep master05 | grep "tgpu.sh" | wc -l)
      if [ $GPU -lt 1 ] ; then

         #if [ $JOBID -ne 0 ] ; then
         #   mv tgpu.sh_master05_"$JOBID".out ../../DL1-OUTPUT/train"$i"_tgpu.sh_master05_"$JOBID".out
         #   mv tgpu.sh_master05_"$JOBID".err ../../DL1-OUTPUT/train"$i"_tgpu.sh_master05_"$JOBID".err
         #fi

         echo "Encolando un nuevo job!!!!"
         sbatch --gres gpu:1 --time=72:00:00 -D ~/work/M3-Project/DL1 tgpu.sh $i
         #sbatch --gres gpu:1 --time=72:00:00 --mem=16000 -D ~/work/M3-Project/DL1 tgpu.sh $i
         #$JOBID=$(squeue | grep master05 | grep "tgpu.sh" | awk 'BEGIN { FS=" " } { print $1 }')
         #echo $pid
         sleep 30
         break         
      else
         #echo "Ejecutando en una GPUs!"
         sleep 30
      fi
   done

done
