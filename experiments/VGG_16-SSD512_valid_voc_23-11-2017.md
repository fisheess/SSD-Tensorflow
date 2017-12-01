# Validate VGG_16-SSD512 on VOC 2007 with batch norm
to see if modular_ssd is really working.
## Training
```bash
python train_ssd_network.py \
    --train_dir=/home/dst/SSD/experiments/vgg_16-ssd512_valid_voc_23-11-2017/logs \
    --dataset_dir=/home/dst/SSD/voc_tfrecords \
    --dataset_name=pascalvoc_2007 \
    --dataset_splict_name=train \
    --model_name=modular_ssd \
    --checkpoint_path=/home/dst/SSD/checkpoints/vgg_16.ckpt \
    --learning_rate=0.1 \
    --learning_rate_decay_type='fixed' \
    --batch_size=12 \
    --feature_extractor=vgg_16 \
    --model=ssd512 \
    --checkpoint_model_scope=vgg_16 \
    --checkpoint_exclude_scopes=ssd512,box_layers \
    --trainable_scopes=ssd512,box_layers \
    --save_summaries_secs=60 \
    --optimizer=adam \
    --num_clones=4 \
    --gpu_momory_fraction=0.9 \
    --weight_decay=0.0005 \
    --max_number_of_steps=30000
```
loss changes very slowly after dropping to 50. Is it possible this is because of the 'feat_block' implementation. Perhaps I should use only one clone to see if this is the problem.
Another strange thing: training the original ssd_512_vgg, regularization loss almost don't change, but mine changed quite a lot.
**A 1-clone test locally**
```bash
python train_ssd_network.py \
    --train_dir=/home/yjin/SSD/experiments/vgg_16-ssd512_valid_voc_23-11-2017/logs \
    --dataset_dir=/media/yjin/Volume/VOC/VOC2007_train_tfrecords \
    --dataset_name=pascalvoc_2007 \
    --dataset_splict_name=train \
    --model_name=modular_ssd \
    --checkpoint_path=/home/yjin/SSD/checkpoints/vgg_16.ckpt \
    --learning_rate=0.1 \
    --learning_rate_decay_type='fixed' \
    --batch_size=12 \
    --feature_extractor=vgg_16 \
    --model=ssd512 \
    --checkpoint_model_scope=vgg_16 \
    --checkpoint_exclude_scopes=ssd512,box_layers \
    --trainable_scopes=ssd512,box_layers \
    --save_summaries_secs=60 \
    --optimizer=adam \
    --num_clones=1 \
    --gpu_momory_fraction=0.9 \
    --weight_decay=0.0005 \
    --max_number_of_steps=30000
```
## Evaluation
```bash
python eval_ssd_network.py \
    --eval_dir=/home/dst/SSD/experiments/vgg_16-ssd512_valid_voc_23-11-2017/eval\
    --dataset_dir=/home/dst/SSD/voc_tfrecords \
    --dataset_name=pascalvoc_2007 \
    --dataset_split_name=test \
    --model_name=modular_ssd \
    --feature_extractor=vgg_16 \
    --model=ssd512 \
    --checkpoint_path=/home/dst/SSD/experiments/vgg_16-ssd512_valid_voc_23-11-2017/logs \
    --wait_for_checkpoints=True \
    --batch_size=1 \
    --max_num_batches=500
```
```bash
python eval_ssd_network.py \
    --eval_dir=/home/yjin/SSD/experiments/vgg_16-ssd512_valid_voc_23-11-2017/eval\
    --dataset_dir=/media/yjin/Volume/VOC/VOC2007_test_tfrecords \
    --dataset_name=pascalvoc_2007 \
    --dataset_split_name=test \
    --model_name=modular_ssd \
    --feature_extractor=vgg_16 \
    --model=ssd512 \
    --checkpoint_path=/home/yjin/SSD/experiments/vgg_16-ssd512_valid_voc_23-11-2017/logs \
    --wait_for_checkpoints=True \
    --batch_size=1 \
    --max_num_batches=500
```
both evaluation gave mAP of 0.001. The model must be buggy.