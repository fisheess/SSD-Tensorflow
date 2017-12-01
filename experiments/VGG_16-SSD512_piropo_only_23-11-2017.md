## Train VGG_16-SSD512 on piropo only, without batch norm (same with original), 23.11.2017
**Training**
```bash
python train_ssd_network.py \
    --train_dir=/home/dst/SSD/piropo_23-11-2017/logs \
    --dataset_dir=/home/dst/SSD/omni_train_tfrecords \
    --dataset_name=piropo \
    --dataset_splict_name=train \
    --model_name=modular_ssd \
    --checkpoint_path=/home/dst/SSD/checkpoints/vgg_16.ckpt \
    --learning_rate=0.005 \
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
    --max_number_of_steps=10000
```
1k steps, lr = 0.01, lr fixed
5k steps, lr = 0.005, lr fixed
10k steps, lr = 0.0001, lr fixed
```bash
python train_ssd_network.py \
    --train_dir=/home/dst/SSD/piropo_23-11-2017/logs \
    --dataset_dir=/home/dst/SSD/omni_train_tfrecords \
    --dataset_name=piropo \
    --dataset_splict_name=train \
    --model_name=modular_ssd \
    --checkpoint_path=/home/dst/SSD/piropo_23-11-2017/model.ckpt-6065 \
    --learning_rate=0.005 \
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
    --max_number_of_steps=10000
```
restart from 6k steps. lr = 0.005
**Evaluation**
```bash
python eval_ssd_network.py \
    --eval_dir=/home/dst/SSD/piropo_23-11-2017/eval\
    --dataset_dir=/home/dst/SSD/omni_test_tfrecords \
    --dataset_name=piropo \
    --dataset_split_name=test \
    --model_name=modular_ssd \
    --feature_extractor=vgg_16 \
    --model=ssd512 \
    --checkpoint_path=/home/dst/SSD/piropo_23-11-2017/logs \
    --wait_for_checkpoints=True \
    --batch_size=1 \
    --max_num_batches=500
```
1k steps, lr = 0.01, loss about 27, mAP = 3.4e-5  
5k steps, lr = 0.005, loss about 20, mAP = 0.0087  
6k steps, lr = 0.0001, loss about 40, mAP = 0.0088  
10k steps, lr = 0.0001, loss about 18, mAP = 8e-5  
This is very strange. After 6k steps I stopped because loss went high again, but test mAP is OK. Then I started again. Loss dropped but mAP dropped too. I tested against training set and test set, both are same.