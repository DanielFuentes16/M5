#!/bin/bash
#SBATCH -n 4 # Number of cores
#SBATCH --mem 8192 # 2GB solicitados.
#SBATCH -p mlow # Partition to submit to
#SBATCH -q masterhigh # Required to requeue other users mlow queue jobs
# With this parameter only 1 job will be running in queue mhigh
# By defaulf the value is masterlow if not defined
#SBATCH --gres gpu:1 # Para pedir Pascales MAX 8
#SBATCH -o %x.out # File to which STDOUT will be written
#SBATCH -e %x.err # File to which STDERR will be written
python3 Mask-CNN_MOTS.py R50-FPN
