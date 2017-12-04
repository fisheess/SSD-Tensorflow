# Fine tune ssd_512_vgg from the pre trained weights by @BalancaP
Since I can't train modular_SSD_tensorflow to the performance of this model, I want to try fine tune the network with
vertical flipping and rotation on person datasets (VOC07+12 person, HDA cam02, PIROPO), so that the net can detect
person on omni images.
## Finetune
```bash
python train_ssd_network.py \
    --train_dir=/home/dst/SSD/experiments/ssd_512_vgg_finetune_04-12-2017/logs \
    --dataset_dir=/home/dst/SSD/person_tfrecords \
    --dataset_name=person \
    --dataset_splict_name=train \
    --model_name=ssd_512_vgg \
    --checkpoint_path=/home/dst/SSD/SSD-Tensorflow/checkpoints/VGG_VOC0712_SSD_512x512_ft_iter_120000.ckpt \
    --learning_rate=0.001 \
    --learning_rate_decay_type='fixed' \
    --batch_size=10 \
    --num_classes=2 \
    --checkpoint_model_scope=vgg_16 \
    --checkpoint_exclude_scopes=ssd_512_vgg/conv6,ssd_512_vgg/conv7,ssd_512_vgg/block8,ssd_512_vgg/block9,ssd_512_vgg/block10,ssd_512_vgg/block11,ssd_512_vgg/block12,ssd_512_vgg/block4_box,ssd_512_vgg/block7_box,ssd_512_vgg/block8_box,ssd_512_vgg/block9_box,ssd_512_vgg/block10_box,ssd_512_vgg/block11_box,ssd_512_vgg/block12_box \
    --trainable_scopes=ssd_512_vgg/conv6,ssd_512_vgg/conv7,ssd_512_vgg/block8,ssd_512_vgg/block9,ssd_512_vgg/block10,ssd_512_vgg/block11,ssd_512_vgg/block12,ssd_512_vgg/block4_box,ssd_512_vgg/block7_box,ssd_512_vgg/block8_box,ssd_512_vgg/block9_box,ssd_512_vgg/block10_box,ssd_512_vgg/block11_box,ssd_512_vgg/block12_box \
    --save_summaries_secs=60 \
    --optimizer=adam \
    --num_clones=4 \
    --gpu_momory_fraction=0.9 \
    --weight_decay=0.0005 \
    --max_number_of_steps=1000
```
## Evaluation
**On PIROPO test**
```bash
python eval_ssd_network.py \
    --eval_dir=/home/dst/SSD/experiments/ssd_512_vgg_finetune_04-12-2017/eval\
    --dataset_dir=/home/dst/SSD/piropo_tfrecords \
    --dataset_name=piropo \
    --num_classes=2 \
    --dataset_split_name=test \
    --model_name=ssd_512_vgg \
    --checkpoint_path=/home/dst/SSD/experiments/ssd_512_vgg_finetune_04-12-2017/logs \
    --wait_for_checkpoints=False \
    --batch_size=1 \
    --max_num_batches=500
```
**On HDA**
```bash
python eval_ssd_network.py \
    --eval_dir=/home/dst/SSD/experiments/ssd_512_vgg_finetune_04-12-2017/eval\
    --dataset_dir=/home/dst/SSD/hda_tfrecords \
    --dataset_name=hda \
    --num_classes=2 \
    --dataset_split_name=test \
    --model_name=ssd_512_vgg \
    --checkpoint_path=/home/dst/SSD/experiments/ssd_512_vgg_finetune_04-12-2017/logs \
    --wait_for_checkpoints=False \
    --batch_size=1 \
    --max_num_batches=500
```