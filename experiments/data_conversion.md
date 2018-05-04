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

```bash
DATASET_DIR=/media/yjin/Volume/PIROPO/test/
OUTPUT_DIR=/media/yjin/Volume/PIROPO/piropo_tfrecords
python tf_convert_data.py \
    --dataset_name=pascalvoc \
    --dataset_dir=${DATASET_DIR} \
    --output_name=piropo_test \
    --output_dir=${OUTPUT_DIR}
```

```bash
DATASET_DIR=/media/yjin/Volume/HDA_dataset/
OUTPUT_DIR=/media/yjin/Volume/HDA_dataset/hda_tfrecords
python tf_convert_data.py \
    --dataset_name=pascalvoc \
    --dataset_dir=${DATASET_DIR} \
    --output_name=hda_train \
    --output_dir=${OUTPUT_DIR}
```

```bash
DATASET_DIR=/media/yjin/Volume/VOC/VOC_person/VOC2012_person/
OUTPUT_DIR=/media/yjin/Volume/VOC/voc_person_tfrecords
python tf_convert_data.py \
    --dataset_name=pascalvoc \
    --dataset_dir=${DATASET_DIR} \
    --output_name=voc12_person_train \
    --output_dir=${OUTPUT_DIR}
```

```bash
DATASET_DIR=/media/yjin/Volume/VOC07_person+HDA+PIROPO_train/
OUTPUT_DIR=/media/yjin/Volume/VOC07_person+HDA+PIROPO_train/person_tfrecords
python tf_convert_data.py \
    --dataset_name=pascalvoc \
    --dataset_dir=${DATASET_DIR} \
    --output_name=person_train \
    --output_dir=${OUTPUT_DIR}
```