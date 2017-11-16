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

## Training
**Train VGG_16-SSD512 on dst-aux-dl**
```bash
python train_ssd_network.py \
    --train_dir=/home/dst/SSD/omni_16-11-2017/logs \
    --dataset_dir=/home/dst/SSD/omni_16-11-2017/tfrecords \
    --dataset_name=omni_train \
    --dataset_splict_name=train \
    --model_name=modular_ssd \
    --checkpoint_path=/home/dst/SSD/checkpoints/vgg_16.ckpt \
    --learning_rate=0.01 \
    --batch_size=24 \
    --feature_extractor=vgg_16 \
    --model=ssd512 \
    --checkpoint_model_scope=vgg_16 \
    --checkpoint_exclude_scopes=ssd512,box_layers \
    --trainable_scopes=ssd512,box_layers \
    --save_summaries_secs=30 \
    --optimizer=adam \
    --num_clones=4 \
    --gpu_momory_fraction=0.9 \
    --weight_decay=0.0005
```