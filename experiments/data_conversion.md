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
