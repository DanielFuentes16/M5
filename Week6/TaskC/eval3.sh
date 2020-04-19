#!/bin/bash
#SBATCH -n 4 # Number of cores
#SBATCH --mem 8192 # 2GB solicitados.
#SBATCH -p mhigh # or mlow Partition to submit to master low prioriy queue
#SBATCH --gres gpu:1 # Para pedir Pascales MAX 8
#SBATCH -o %x.out # File to which STDOUT will be written
#SBATCH -e %x.err # File to which STDERR will be written
python3 deeplab/eval.py \
    --logtostderr \
    --eval_split="val_fine" \
    --model_variant="xception_65" \
    --atrous_rates=6 \
    --atrous_rates=12 \
    --atrous_rates=18 \
    --output_stride=16 \
    --decoder_output_stride=4 \
    --eval_crop_size="1025,2049" \
    --dataset="cityscapes" \
    --checkpoint_dir=/home/grupo09/marc/m5-w6/week6/taskc/x65-logdir-3 \
    --eval_logdir=/home/grupo09/marc/m5-w6/week6/taskc/x65-logdir-eval-3 \
    --dataset_dir=/home/grupo09/marc/m5-w6/week6/taskc/models/research/deeplab/datasets/cityscapes/tfrecord
