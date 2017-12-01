# Commands for modular_ssd
## Convert data to tf records
**Convert VOC dataset to tf records**
```bash
DATASET_DIR=/home/yjin/data/VOC/VOC2007/trainval/
OUTPUT_DIR=/home/yjin/data/VOC/tfrecords/VOC2007_trainval

DATASET_DIR=/home/yjin/data/VOC/VOC2007/test/
OUTPUT_DIR=/home/yjin/data/VOC/tfrecords/VOC2007_test

DATASET_DIR=/home/yjin/data/VOC/VOC2012/trainval/
OUTPUT_DIR=/home/yjin/data/VOC/tfrecords/VOC2012_trainval

python tf_convert_data.py \
    --dataset_name=pascalvoc \
    --dataset_dir=${DATASET_DIR} \
    --output_name=voc_2012_train \
    --output_dir=${OUTPUT_DIR}
```
## Training
**Train VGG_16-SSD300**
```bash
python train_ssd_network.py \
    --train_dir=/home/yjin/SSD/training/logs/tmp \
    --dataset_dir=/home/yjin/data/VOC/tfrecords/VOC2007_trainval \
    --dataset_name=pascalvoc_2007 \
    --dataset_splict_name=train \
    --model_name=modular_ssd \
    --checkpoint_path=/home/yjin/SSD/checkpoints/vgg_16.ckpt \
    --learning_rate=0.01 \
    --batch_size=24 \
    --feature_extractor=vgg_16 \
    --model=ssd300 \
    --num_classes=21 \
    --checkpoint_model_scope=vgg_16 \
    --checkpoint_exclude_scopes=ssd300,box_layers \
    --trainable_scopes=ssd300,box_layers \
    --save_summaries_secs=30 \
    --optimizer=adam 
```

**Train VGG_16-SSD300 on dst-aux-dl**
```bash
python train_ssd_network.py \
    --train_dir=/home/dst/SSD/logs/vgg_16-ssd300 \
    --dataset_dir=/home/dst/SSD/voc2007tf \
    --dataset_name=pascalvoc_2007 \
    --dataset_splict_name=train \
    --model_name=modular_ssd \
    --checkpoint_path=/home/dst/SSD/checkpoints/vgg_16.ckpt \
    --learning_rate=0.01 \
    --batch_size=24 \
    --feature_extractor=vgg_16 \
    --model=ssd300 \
    --checkpoint_model_scope=vgg_16 \
    --checkpoint_exclude_scopes=ssd300,box_layers \
    --trainable_scopes=ssd300,box_layers \
    --save_summaries_secs=30 \
    --optimizer=adam \
    --num_clones=4 \
    --gpu_momory_fraction=0.9 \
    --weight_decay=0.0005
```

**Train Resnet_V1_50-SSD512 on dst-aux-dl**
```bash
python train_ssd_network.py \
    --train_dir=/home/dst/SSD/logs/Resnet_V1_50-SSD512 \
    --dataset_dir=/home/dst/SSD/voc2007tf \
    --dataset_name=pascalvoc_2007 \
    --dataset_splict_name=train \
    --model_name=modular_ssd \
    --checkpoint_path=/home/dst/SSD/checkpoints/vgg_16.ckpt \
    --learning_rate=0.01 \
    --batch_size=24 \
    --feature_extractor=vgg_16 \
    --model=ssd300 \
    --checkpoint_model_scope=vgg_16 \
    --checkpoint_exclude_scopes=ssd300,box_layers \
    --trainable_scopes=ssd300,box_layers \
    --save_summaries_secs=30 \
    --optimizer=adam \
    --num_clones=4 \
    --gpu_momory_fraction=0.9 \
    --weight_decay=0.0005
```

