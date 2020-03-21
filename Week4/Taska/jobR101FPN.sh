#!/bin/bash
#SBATCH -n 4 # Number of cores
#SBATCH --mem 8192 # 2GB solicitados.
#SBATCH -p mhigh,mhigh # or mlow Partition to submit to master low prioriy queue
#SBATCH --gres gpu:1 # Para pedir Pascales MAX 8
#SBATCH -o %x.out # File to which STDOUT will be written
#SBATCH -e %x.err # File to which STDERR will be written
python3 Mask-CNN_MOTS.py R101-FPN
