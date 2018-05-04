## Training on 16.11.2017
**Train VGG_16-SSD512 on dst-aux-dl**
first 35k steps
```bash
python train_ssd_network.py \
    --train_dir=/home/dst/SSD/omni_16-11-2017/logs \
    --dataset_dir=/home/dst/SSD/omni_16-11-2017/tfrecords \
    --dataset_name=omni \
    --dataset_splict_name=train \
    --model_name=modular_ssd \
    --checkpoint_path=/home/dst/SSD/checkpoints/vgg_16.ckpt \
    --learning_rate=0.01 \ 
    --batch_size=12 \
    --feature_extractor=vgg_16 \
    --model=ssd512 \
    --checkpoint_model_scope=vgg_16 \
    --checkpoint_exclude_scopes=ssd512,box_layers \
    --trainable_scopes=ssd512,box_layers \
    --save_summaries_secs=30 \
    --optimizer=adam \
    --num_clones=4 \
    --gpu_momory_fraction=0.9 \
    --weight_decay=0.0005 \
    --max_number_of_steps=90000
```
Loss stopped dropping, try raising LR
```bash
python train_ssd_network.py \
    --train_dir=/home/dst/SSD/omni_16-11-2017/logs \
    --dataset_dir=/home/dst/SSD/omni_16-11-2017/tfrecords \
    --dataset_name=omni \
    --dataset_splict_name=train \
    --model_name=modular_ssd \
    --checkpoint_path=/home/dst/SSD/checkpoints/vgg_16.ckpt \
    --learning_rate=0.01 \
    --learning_rate_decay_type='fixed' \
    --batch_size=12 \
    --feature_extractor=vgg_16 \
    --model=ssd512 \
    --checkpoint_model_scope=vgg_16 \
    --checkpoint_exclude_scopes=ssd512,box_layers \
    --trainable_scopes=ssd512,box_layers \
    --save_summaries_secs=30 \
    --optimizer=adam \
    --num_clones=4 \
    --gpu_momory_fraction=0.9 \
    --weight_decay=0.0005 \
    --max_number_of_steps=90000
```
### Evaluation
**Test on VOC2007 test**
```bash
python eval_ssd_network.py \
    --eval_dir=/home/dst/SSD/omni_16-11-2017/eval\
    --dataset_dir=/home/dst/SSD/VOC2007_test_tfrecords \
    --dataset_name=pascalvoc_2007 \
    --dataset_split_name=test \
    --model_name=modular_ssd \
    --feature_extractor=vgg_16 \
    --model=ssd512 \
    --checkpoint_path=/home/dst/SSD/omni_16-11-2017/logs \
    --wait_for_checkpoints=True \
    --batch_size=1 \
    --max_num_batches=500
```

Evaluation very bad: mAP = 0.052

Training seems not have converged. No idea why. Will try using native ssd512 by Paul Balanca.