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
