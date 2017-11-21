import tensorflow as tf
from datasets import pascalvoc_common

slim = tf.contrib.slim

FILE_PATTERN = 'omni_%s_*.tfrecord'
ITEMS_TO_DESCRIPTIONS = {
    'image': 'A color image of varying height and width.',
    'shape': 'Shape of the image',
    'object/bbox': 'A list of bounding boxes, one per each object.',
    'object/label': 'A list of labels, one per each object.',
}
# (Images, Objects) statistics on every class.
TRAIN_STATISTICS = {
    'none': (0, 0),
    'aeroplane': (0, 0),
    'bicycle': (0, 0),
    'bird': (0, 0),
    'boat': (0, 0),
    'bottle': (0, 0),
    'bus': (0, 0),
    'car': (0, 0),
    'cat': (0, 0),
    'chair': (0, 0),
    'cow': (0, 0),
    'diningtable': (0, 0),
    'dog': (0, 0),
    'horse': (0, 0),
    'motorbike': (0, 0),
    'person': (4832, 4840),
    'pottedplant': (0, 0),
    'sheep': (0, 0),
    'sofa': (0, 0),
    'train': (0, 0),
    'tvmonitor': (0, 0),
    'total': (4832, 4840),
}
TEST_STATISTICS = {
    'none': (0, 0),
    'aeroplane': (1, 1),
    'bicycle': (1, 1),
    'bird': (1, 1),
    'boat': (1, 1),
    'bottle': (1, 1),
    'bus': (1, 1),
    'car': (1, 1),
    'cat': (1, 1),
    'chair': (1, 1),
    'cow': (1, 1),
    'diningtable': (1, 1),
    'dog': (1, 1),
    'horse': (1, 1),
    'motorbike': (1, 1),
    'person': (1, 1),
    'pottedplant': (1, 1),
    'sheep': (1, 1),
    'sofa': (1, 1),
    'train': (1, 1),
    'tvmonitor': (1, 1),
    'total': (20, 20),
}
SPLITS_TO_SIZES = {
    'train': 2409,
    'test': 2423
}
SPLITS_TO_STATISTICS = {
    'train': TRAIN_STATISTICS,
    'test': TEST_STATISTICS
}
NUM_CLASSES = 20


def get_split(split_name, dataset_dir, file_pattern=None, reader=None):
    """Gets a dataset tuple with instructions for reading ImageNet.

    Args:
      split_name: A train/test split name.
      dataset_dir: The base directory of the dataset sources.
      file_pattern: The file pattern to use when matching the dataset sources.
        It is assumed that the pattern contains a '%s' string so that the split
        name can be inserted.
      reader: The TensorFlow reader type.

    Returns:
      A `Dataset` namedtuple.

    Raises:
        ValueError: if `split_name` is not a valid train/test split.
    """
    if not file_pattern:
        file_pattern = FILE_PATTERN
    return pascalvoc_common.get_split(split_name, dataset_dir,
                                      file_pattern, reader,
                                      SPLITS_TO_SIZES,
                                      ITEMS_TO_DESCRIPTIONS,
                                      NUM_CLASSES)