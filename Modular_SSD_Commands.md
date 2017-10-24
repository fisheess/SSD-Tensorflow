**Convert VOC dataset to tf records**
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

**Train VGG_16-SSD300**
```bash
$ DATASET_DIR = 
$ TRAIN_DIR = ./logs
$ CHECKPOINT_PATH =./checkpoints/vgg.ckpt
$ python train_ssd_network.py \
$     --train_dir=${TRAIN_DIR} \
$     --dataset_dir=${DATASET_DIR} \
$     --dataset_name= 
```
