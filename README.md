Week 5: Introduction to Object Detection 
## Group 3
### Keyao Li ryelesssi@gmail.com
### Marc marc.oros@e-campus.uab.cat
### Daniel Fuentes daanyfuentes@gmail.com
### [Overleaf Report](https://www.overleaf.com/read/djsbfhpnqrqy "Overleaf")
### [Reduced Overleaf Report in CVPR format](https://www.overleaf.com/read/zxfqkmdjxwbm "Overleaf")

### [Team 3 Final Presentation](https://drive.google.com/file/d/16BY1adIj2M8w-ts9S3_6PGbf2ruWarIF/view?usp=sharing "Team3 Presentation")

## To run the code
To run each task run the scripts located in the Week6/TaskX folder as follows:


### Task A
`python3 Mask-CNN_MOTS.py R50-FPN
If only is necessary to ejecute concrete data augmentation is necesary to change data_it in the Mask-CNN_MOTS.py
### Task B
To run the three configurations for experiment 1 run:
<pre><code>python3 Experiment1.py
python3 Experiment2.py
python3 Experiment3.py</code></pre>

To run the different combinations for experiment 2 run:\
(for 80% of real dataset)
<pre><code>python3 Experiment4.py 0.8</code></pre>

If you want to remove n sequences from the real dataset\
(to remove 3 sequences from the real dataset)
<pre><code>python3 Experiment4.py 3 -s</code></pre>

### Task C
The process to run Task c is slightly more involved

1. Clone tensorflow's models repository, which includes Deeplabv3+
<pre><code>git clone https://github.com/tensorflow/models.git</code></pre>

2. Convert the cityscapes dataset to tfrecord format
    
<pre><code># From the tensorflow/models/research/deeplab/datasets directory.
sh convert_cityscapes.sh</code></pre>

3. Use the following command to train the model:

<pre><code>python3 deeplab/train.py \
    --logtostderr \
    --training_number_of_steps=50000 \
    --train_split="train_fine" \
    --model_variant="xception_65" \
    --atrous_rates=6 \
    --atrous_rates=12 \
    --atrous_rates=18 \
    --output_stride=16 \
    --train_crop_size="769,769" \
    --train_batch_size=2 \
    --add_image_level=True \
    --dataset="cityscapes" \
    --tf_initial_checkpoint=${INITIAL_CHECKPOINT} \
    --train_logdir=${LOGDIR_PATH} \
    --dataset_dir=${TFRECORD_DIR}</code></pre>

4. Use the following command to evaluate the model's performance:

<pre><code>python3 deeplab/eval.py \
    --logtostderr \
    --eval_split="val_fine" \
    --model_variant="xception_65" \
    --atrous_rates=6 \
    --atrous_rates=12 \
    --atrous_rates=18 \
    --output_stride=16 \
    --add_image_level=True \
    --eval_crop_size="1025,2049" \
    --dataset="cityscapes" \
    --checkpoint_dir=${LOGDIR_PATH} \
    --eval_logdir=${EVAL_PATH} \
    --dataset_dir=${TFRECORD_DIR}</code></pre>

5. Use the following command to visualize the results of the model on the dataset:

<pre><code>python3 deeplab/vis.py \
    --logtostderr \
    --vis_split="val_fine" \
    --model_variant="xception_65" \
    --atrous_rates=6 \
    --atrous_rates=12 \
    --atrous_rates=18 \
    --output_stride=16 \
    --add_image_level=True \
    --vis_crop_size="1025,2049" \
    --dataset="cityscapes" \
    --colormap_type="cityscapes" \
    --checkpoint_dir=${LOGDIR_PATH} \
    --vis_logdir=${VIS_PATH} \
    --dataset_dir=${TFRECORD_DIR}</code></pre>

The three previous commands are for the first experiment in table 7 of the paper, for the other three models refer to the `week6/TaskC` folder to see the rest of the scripts
