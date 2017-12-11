# Fine tune ssd_512_vgg from the pre trained weights by @BalancaP
Since I can't train modular_SSD_tensorflow to the performance of this model, I want to try fine tune the network with
vertical flipping and rotation on person datasets (VOC07+12 person, HDA cam02, PIROPO), so that the net can detect
person on omni images.
## Finetune
###First trial
```bash
python train_ssd_network.py \
    --train_dir=/home/dst/SSD/experiments/ssd_512_vgg_finetune_04-12-2017/logs \
    --dataset_dir=/home/dst/SSD/person_tfrecords \
    --dataset_name=person \
    --dataset_splict_name=train \
    --model_name=ssd_512_vgg \
    --checkpoint_path=/home/dst/SSD/SSD-Tensorflow/checkpoints/VGG_VOC0712_SSD_512x512_ft_iter_120000.ckpt \
    --learning_rate=0.0005 \
    --learning_rate_decay_type='fixed' \
    --batch_size=8 \
    --save_summaries_secs=60 \
    --save_interval_secs=3600 \
    --optimizer=adam \
    --num_clones=4 \
    --gpu_momory_fraction=0.9 \
    --weight_decay=0.0005 \
    --max_to_keep=20 \
    --max_number_of_steps=150000
```
The orginal weights are trained on VOC07+12 trainval. It should be expected, that the finetuning will not converge very
fast. For one, the dataset are quite mixed. For another, flipping and rotation makes the images very different from the 
original training dataset.
Tuned down lr to 0.0005 and start from 67656 steps again. At 150k steps, loss = ~10
###Try higher lr
```bash
python train_ssd_network.py \
    --train_dir=/home/dst/SSD/experiments/ssd_512_vgg_finetune_04-12-2017/logs1 \
    --dataset_dir=/home/dst/SSD/person_tfrecords \
    --dataset_name=person \
    --dataset_splict_name=train \
    --model_name=ssd_512_vgg \
    --checkpoint_path=/home/dst/SSD/experiments/ssd_512_vgg_finetune_04-12-2017/logs \
    --learning_rate=0.01 \
    --learning_rate_decay_type='fixed' \
    --batch_size=8 \
    --save_summaries_secs=60 \
    --save_interval_secs=3600 \
    --optimizer=adam \
    --num_clones=4 \
    --gpu_momory_fraction=0.9 \
    --weight_decay=0.0005 \
    --max_number_of_steps=8000
```
use the weights from the first phase of fine tuning to start experiments with higher lr. 

## Evaluation
###On training set
#### Pre-trained weights on Pascal VOC07+12 trainval, starting point of fine tuning
```bash
python eval_ssd_network.py \
    --eval_dir=/home/dst/SSD/experiments/ssd_512_vgg_finetune_04-12-2017/eval\
    --dataset_dir=/home/dst/SSD/person_tfrecords \
    --dataset_name=person \
    --dataset_split_name=train \
    --model_name=ssd_512_vgg \
    --checkpoint_path=/home/dst/SSD/SSD-Tensorflow/checkpoints/VGG_VOC0712_SSD_512x512_ft_iter_120000.ckpt \
    --wait_for_checkpoints=False \
    --batch_size=1 \
    --max_num_batches=500
```
mAP = 0.0338
####After fine tuning
```bash
python eval_ssd_network.py \
    --eval_dir=/home/dst/SSD/experiments/ssd_512_vgg_finetune_04-12-2017/eval\
    --dataset_dir=/home/dst/SSD/person_tfrecords \
    --dataset_name=person \
    --dataset_split_name=train \
    --model_name=ssd_512_vgg \
    --checkpoint_path=/home/dst/SSD/experiments/ssd_512_vgg_finetune_04-12-2017/logs \
    --wait_for_checkpoints=True \
    --batch_size=1 \
    --max_num_batches=500
```
First trial, 10k steps, mAP = 0.0429
20k steps, mAP = 0.0432
32k steps, mAP = 0.0443
43.8k steps, mAP = 0.0442
59.7k steps, mAP = 0.0444
67.6k steps, mAP = 0.0444
80k steps, mAP = 0.0446
150k steps, mAP = 0.0451
```bash
python eval_ssd_network.py \
    --eval_dir=/home/dst/SSD/experiments/ssd_512_vgg_finetune_04-12-2017/eval1\
    --dataset_dir=/home/dst/SSD/person_tfrecords \
    --dataset_name=person \
    --dataset_split_name=train \
    --model_name=ssd_512_vgg \
    --checkpoint_path=/home/dst/SSD/experiments/ssd_512_vgg_finetune_04-12-2017/logs1 \
    --wait_for_checkpoints=False \
    --batch_size=1 \
    --max_num_batches=500
```
Second trial, 3k steps, mAP = 0.0376
8k steps, mAP = 0.0385
13k steps, mAP = 0.0387
###On PIROPO test
#### Pre-trained weights on Pascal VOC07+12 trainval, starting point of fine tuning
```bash
python eval_ssd_network.py \
    --eval_dir=/home/dst/SSD/experiments/ssd_512_vgg_finetune_04-12-2017/eval\
    --dataset_dir=/home/dst/SSD/piropo_tfrecords \
    --dataset_name=piropo \
    --dataset_split_name=test \
    --model_name=ssd_512_vgg \
    --checkpoint_path=/home/dst/SSD/SSD-Tensorflow/checkpoints/VGG_VOC0712_SSD_512x512_ft_iter_120000.ckpt \
    --wait_for_checkpoints=False \
    --batch_size=1 \
    --max_num_batches=500
```
mAP = 0.0190
####After fine tuning
```bash
python eval_ssd_network.py \
    --eval_dir=/home/dst/SSD/experiments/ssd_512_vgg_finetune_04-12-2017/eval\
    --dataset_dir=/home/dst/SSD/piropo_tfrecords \
    --dataset_name=piropo \
    --dataset_split_name=test \
    --model_name=ssd_512_vgg \
    --checkpoint_path=/home/dst/SSD/experiments/ssd_512_vgg_finetune_04-12-2017/logs \
    --wait_for_checkpoints=False \
    --batch_size=1 \
    --max_num_batches=500
```
First trial, 10k steps, mAP = 0.0454
20k steps, mAP = 0.0469
59.7k steps, mAP = 0.0454
67.6k steps, mAP = 0.0472
80k steps, mAP = 0.0454
150k steps, mAP = 0.0454
```bash
python eval_ssd_network.py \
    --eval_dir=/home/dst/SSD/experiments/ssd_512_vgg_finetune_04-12-2017/eval1\
    --dataset_dir=/home/dst/SSD/piropo_tfrecords \
    --dataset_name=piropo \
    --dataset_split_name=test \
    --model_name=ssd_512_vgg \
    --checkpoint_path=/home/dst/SSD/experiments/ssd_512_vgg_finetune_04-12-2017/logs1 \
    --wait_for_checkpoints=False \
    --batch_size=1 \
    --max_num_batches=500
```
Second trial, 3k steps, mAP = 0.0454
8k steps, mAP = 0.0453
13k steps, mAP = 0.0454
###On HDA
#### Pre-trained weights on Pascal VOC07+12 trainval, starting point of fine tuning
```bash
python eval_ssd_network.py \
    --eval_dir=/home/dst/SSD/experiments/ssd_512_vgg_finetune_04-12-2017/eval\
    --dataset_dir=/home/dst/SSD/hda_tfrecords \
    --dataset_name=hda \
    --dataset_split_name=train \
    --model_name=ssd_512_vgg \
    --checkpoint_path=/home/dst/SSD/SSD-Tensorflow/checkpoints/VGG_VOC0712_SSD_512x512_ft_iter_120000.ckpt \
    --wait_for_checkpoints=False \
    --batch_size=1 \
    --max_num_batches=500
```
mAP = 0.0283
####After fine tuning
```bash
python eval_ssd_network.py \
    --eval_dir=/home/dst/SSD/experiments/ssd_512_vgg_finetune_04-12-2017/eval\
    --dataset_dir=/home/dst/SSD/hda_tfrecords \
    --dataset_name=hda \
    --dataset_split_name=train \
    --model_name=ssd_512_vgg \
    --checkpoint_path=/home/dst/SSD/experiments/ssd_512_vgg_finetune_04-12-2017/logs \
    --wait_for_checkpoints=False \
    --batch_size=1 \
    --max_num_batches=500
```
First trial, 10k steps, mAP = 0.0451
20k steps, mAP = 0.0454
59.7k steps, mAP = 0.0454
67.6k steps, mAP = 0.0454
80k steps, mAP = 0.0454
150 k steps, mAP = 0.0454
```bash
python eval_ssd_network.py \
    --eval_dir=/home/dst/SSD/experiments/ssd_512_vgg_finetune_04-12-2017/eval1\
    --dataset_dir=/home/dst/SSD/hda_tfrecords \
    --dataset_name=hda \
    --dataset_split_name=train \
    --model_name=ssd_512_vgg \
    --checkpoint_path=/home/dst/SSD/experiments/ssd_512_vgg_finetune_04-12-2017/logs1 \
    --wait_for_checkpoints=False \
    --batch_size=1 \
    --max_num_batches=500
```
Second trial, 3k steps, mAP = 0.0439
8k steps, mAP = 0.0447
13k steps, mAP =0.0451