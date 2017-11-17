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
    'aeroplane': (670+238, 865+306),
    'bicycle': (552+243, 711+353),
    'bird': (765+330, 1119+486),
    'boat': (508+181, 850+290),
    'bottle': (706+244, 1259+505),
    'bus': (421+186, 593+229),
    'car': (1161+713, 2017+1250),
    'cat': (1080+337, 1217+376),
    'chair': (1119+445, 2354+798),
    'cow': (303+141, 588+259),
    'diningtable': (538+200, 609+215),
    'dog': (1286+421, 1515+510),
    'horse': (482+287, 710+362),
    'motorbike': (526+245, 713+339),
    'person': (4087+2008+2423+1388, 8566+4690+2427+1832),
    'pottedplant': (527+245, 973+514),
    'sheep': (325+96, 813+257),
    'sofa': (507+229, 566+248),
    'train': (544+261, 628+297),
    'tvmonitor': (575+256, 784+324),
    'total': (11540+5011, 27450+12608),
}
SPLITS_TO_SIZES = {
    'train': 25947,
}
SPLITS_TO_STATISTICS = {
    'train': TRAIN_STATISTICS,
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