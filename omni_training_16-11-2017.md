## Data conversion
```bash
DATASET_DIR=/media/yjin/Volume/Omni_Training_16-11-2017/
OUTPUT_DIR=/media/yjin/Volume/Omni_Training_16-11-2017/tfrecords
python tf_convert_data.py \
    --dataset_name=pascalvoc \
    --dataset_dir=${DATASET_DIR} \
    --output_name=omni_train \
    --output_dir=${OUTPUT_DIR}
```

```bash
DATASET_DIR=/media/yjin/Volume/PIROPO/testsampled/
OUTPUT_DIR=/media/yjin/Volume/PIROPO/testsampled/tfrecords
python tf_convert_data.py \
    --dataset_name=pascalvoc \
    --dataset_dir=${DATASET_DIR} \
    --output_name=omni_test \
    --output_dir=${OUTPUT_DIR}
```

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

## Original ssd 512 by Paul Balanca, 20.11.2017
**Training ssd_512_vgg**
```bash
python train_ssd_network.py \
    --train_dir=/home/dst/SSD/omni_20-11-2017_ssd_512_vgg/logs \
    --dataset_dir=/home/dst/SSD/tfrecords \
    --dataset_name=omni \
    --dataset_splict_name=train \
    --model_name=ssd_512_vgg \
    --checkpoint_path=/home/dst/SSD/checkpoints/vgg_16.ckpt \
    --learning_rate=0.001 \
    --learning_rate_decay_factor=0.94 \
    --batch_size=12 \
    --checkpoint_model_scope=vgg_16 \
    --checkpoint_exclude_scopes=ssd_512_vgg/conv6,ssd_512_vgg/conv7,ssd_512_vgg/block8,ssd_512_vgg/block9,ssd_512_vgg/block10,ssd_512_vgg/block11,ssd_512_vgg/block12,ssd_512_vgg/block4_box,ssd_512_vgg/block7_box,ssd_512_vgg/block8_box,ssd_512_vgg/block9_box,ssd_512_vgg/block10_box,ssd_512_vgg/block11_box,ssd_512_vgg/block12_box \
    --trainable_scopes=ssd_512_vgg/conv6,ssd_512_vgg/conv7,ssd_512_vgg/block8,ssd_512_vgg/block9,ssd_512_vgg/block10,ssd_512_vgg/block11,ssd_512_vgg/block12,ssd_512_vgg/block4_box,ssd_512_vgg/block7_box,ssd_512_vgg/block8_box,ssd_512_vgg/block9_box,ssd_512_vgg/block10_box,ssd_512_vgg/block11_box,ssd_512_vgg/block12_box \
    --save_summaries_secs=30 \
    --optimizer=adam \
    --num_clones=4 \
    --gpu_momory_fraction=0.9 \
    --weight_decay=0.0005 \
    --max_number_of_steps=90000
``` 
**Evaluation**
```bash
python eval_ssd_network.py \
    --eval_dir=/home/dst/SSD/omni_20-11-2017_ssd_512_vgg/eval\
    --dataset_dir=/home/dst/SSD/VOC2007_test_tfrecords \
    --dataset_name=pascalvoc_2007 \
    --dataset_split_name=test \
    --model_name=ssd_512_vgg \
    --checkpoint_path=/home/dst/SSD/omni_20-11-2017_ssd_512_vgg/logs \
    --wait_for_checkpoints=True \
    --batch_size=1 \
    --max_num_batches=500
```

```bash
python eval_ssd_network.py \
    --eval_dir=/home/dst/SSD/omni_20-11-2017_ssd_512_vgg/eval\
    --dataset_dir=/home/dst/SSD/omni_test_tfrecords \
    --dataset_name=omni \
    --dataset_split_name=test \
    --model_name=ssd_512_vgg \
    --checkpoint_path=/home/dst/SSD/omni_20-11-2017_ssd_512_vgg/logs \
    --wait_for_checkpoints=True \
    --batch_size=1 \
    --max_num_batches=500
```
At 2136 epochs mAP is already at 0.055. I will train another 2k epochs to see if it improves.

At 4279 epochs mAP is 0.105. I think there is some problem with modular_ssd.

Let training run some more time. Try to find the problem with modular_ssd at the same time.

After lunch, 8614 epochs, mAP = 0.198

## New training on 20.11
**Train VGG_16-SSD512 on dst-aux-dl**
```bash
python train_ssd_network.py \
    --train_dir=/home/dst/SSD/omni_20-11-2017/logs \
    --dataset_dir=/home/dst/SSD/tfrecords \
    --dataset_name=omni \
    --dataset_splict_name=train \
    --model_name=modular_ssd \
    --checkpoint_path=/home/dst/SSD/checkpoints/vgg_16.ckpt \
    --learning_rate=0.002 \
    --batch_size=10 \
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
    --eval_dir=/home/dst/SSD/omni_20-11-2017/eval\
    --dataset_dir=/home/dst/SSD/VOC2007_test_tfrecords \
    --dataset_name=pascalvoc_2007 \
    --dataset_split_name=test \
    --model_name=modular_ssd \
    --feature_extractor=vgg_16 \
    --model=ssd512 \
    --checkpoint_path=/home/dst/SSD/omni_20-11-2017/logs \
    --wait_for_checkpoints=True \
    --batch_size=1 \
    --max_num_batches=500
```

## Training ssd_512_vgg on PIROPO only, 21.11.2017
**Training**
```bash
python train_ssd_network.py \
    --train_dir=/home/dst/SSD/piropo_21-11-2017/logs \
    --dataset_dir=/home/dst/SSD/omni_train_tfrecords \
    --dataset_name=piropo \
    --dataset_splict_name=train \
    --model_name=ssd_512_vgg \
    --checkpoint_path=/home/dst/SSD/checkpoints/vgg_16.ckpt \
    --learning_rate=0.001 \
    --learning_rate_decay_factor=0.94 \
    --batch_size=12 \
    --checkpoint_model_scope=vgg_16 \
    --checkpoint_exclude_scopes=ssd_512_vgg/conv6,ssd_512_vgg/conv7,ssd_512_vgg/block8,ssd_512_vgg/block9,ssd_512_vgg/block10,ssd_512_vgg/block11,ssd_512_vgg/block12,ssd_512_vgg/block4_box,ssd_512_vgg/block7_box,ssd_512_vgg/block8_box,ssd_512_vgg/block9_box,ssd_512_vgg/block10_box,ssd_512_vgg/block11_box,ssd_512_vgg/block12_box \
    --trainable_scopes=ssd_512_vgg/conv6,ssd_512_vgg/conv7,ssd_512_vgg/block8,ssd_512_vgg/block9,ssd_512_vgg/block10,ssd_512_vgg/block11,ssd_512_vgg/block12,ssd_512_vgg/block4_box,ssd_512_vgg/block7_box,ssd_512_vgg/block8_box,ssd_512_vgg/block9_box,ssd_512_vgg/block10_box,ssd_512_vgg/block11_box,ssd_512_vgg/block12_box \
    --save_summaries_secs=30 \
    --optimizer=adam \
    --num_clones=4 \
    --gpu_momory_fraction=0.9 \
    --weight_decay=0.0005 \
    --max_number_of_steps=90000
``` 
```bash
python train_ssd_network.py \
    --train_dir=/home/dst/SSD/piropo_21-11-2017/logs \
    --dataset_dir=/home/dst/SSD/omni_train_tfrecords \
    --dataset_name=piropo \
    --dataset_splict_name=train \
    --model_name=ssd_512_vgg \
    --checkpoint_path=/home/dst/SSD/checkpoints/vgg_16.ckpt \
    --learning_rate=0.001 \
    --learning_rate_decay_factor=0.94 \
    --batch_size=12 \
    --checkpoint_model_scope=vgg_16 \
    --checkpoint_exclude_scopes=ssd_512_vgg/conv6,ssd_512_vgg/conv7,ssd_512_vgg/block8,ssd_512_vgg/block9,ssd_512_vgg/block10,ssd_512_vgg/block11,ssd_512_vgg/block12,ssd_512_vgg/block4_box,ssd_512_vgg/block7_box,ssd_512_vgg/block8_box,ssd_512_vgg/block9_box,ssd_512_vgg/block10_box,ssd_512_vgg/block11_box,ssd_512_vgg/block12_box \
    --save_summaries_secs=30 \
    --optimizer=adam \
    --num_clones=4 \
    --gpu_momory_fraction=0.9 \
    --weight_decay=0.0005 \
    --max_number_of_steps=90000
``` 
**Test on piropo**
```bash
python eval_ssd_network.py \
    --eval_dir=/home/dst/SSD/piropo_21-11-2017/eval\
    --dataset_dir=/home/dst/SSD/omni_test_tfrecords \
    --dataset_name=piropo \
    --dataset_split_name=test \
    --model_name=ssd_512_vgg \
    --checkpoint_path=/home/dst/SSD/piropo_21-11-2017/logs \
    --wait_for_checkpoints=True \
    --batch_size=1 \
    --max_num_batches=500
```
4k steps, mAP = 0.024  
9k steps, mAP = 0.03  
15k steps, mAP = 0.0326  
22k steps, mAP = 0.0327  
27k steps, mAP = 0.0329  
35k steps, mAP = 0.03297  
42k steps, mAP = 0.033  
50k steps, mAP = 0.033  
100k steps, mAP = 0.033  
146k steps, mAP = 0.033  
159k steps, mAP = 0.033  
268k steps, mAP = 0.033  
So it seems to be improving in the first 40k steps or so. Checked the learning rate, I have forgot to set the min_learning_rate. Still, the result is not optimal. 

## Train VGG_16-SSD512 on piropo only, without batch norm (same with original), 23.11.2017
