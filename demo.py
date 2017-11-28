from pathlib import Path
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from nets import modular_ssd, np_methods, ssd_vgg_512
from preprocessing import ssd_vgg_preprocessing
from notebooks import visualization


# TensorFlow session: grow memory when needed.
gpu_options = tf.GPUOptions(allow_growth=True)
config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
isess = tf.InteractiveSession(config=config)

# Input placeholder.
net_shape = (512, 512)
data_format = 'NHWC'
img_input = tf.placeholder(tf.uint8, shape=(None, None, 3))
# Evaluation pre-processing: resize to SSD net shape.
image_pre, labels_pre, bboxes_pre, bbox_img = ssd_vgg_preprocessing.preprocess_for_eval(
    img_input, None, None, net_shape, data_format, resize=ssd_vgg_preprocessing.Resize.WARP_RESIZE)
image_4d = tf.expand_dims(image_pre, 0)

# Define the SSD model.
demo_params = ssd_vgg_512.SSDParams(
    img_shape=(512, 512),
    num_classes=2,
    no_annotation_label=21,
    feat_layers=['block4', 'block7', 'block8', 'block9', 'block10', 'block11', 'block12'],
    feat_shapes=[(64, 64), (32, 32), (16, 16), (8, 8), (4, 4), (2, 2), (1, 1)],
    anchor_size_bounds=[0.10, 0.90],
    anchor_sizes=[(20.48, 51.2),
                  (51.2, 133.12),
                  (133.12, 215.04),
                  (215.04, 296.96),
                  (296.96, 378.88),
                  (378.88, 460.8),
                  (460.8, 542.72)],
    anchor_ratios=[[2, .5],
                   [2, .5, 3, 1./3],
                   [2, .5, 3, 1./3],
                   [2, .5, 3, 1./3],
                   [2, .5, 3, 1./3],
                   [2, .5],
                   [2, .5]],
    anchor_steps=[8, 16, 32, 64, 128, 256, 512],
    anchor_offset=0.5,
    normalizations=[20, -1, -1, -1, -1, -1, -1],
    prior_scaling=[0.1, 0.1, 0.2, 0.2]
)
ssd_net = ssd_vgg_512.SSDNet(demo_params)
predictions, localisations, _, _ = ssd_net.net(image_4d, is_training=False)
# Restore SSD model.
ckpt_filename = '/home/yjin/SSD/experiments/logs/model.ckpt-10000'
isess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(isess, ckpt_filename)
# SSD default anchor boxes.
ssd_anchors = ssd_net.anchors(net_shape)


# Main image processing routine.
def process_image(img, select_threshold=0.5, nms_threshold=.45, net_shape=(512, 512)):
    # Run SSD network.
    rimg, rpredictions, rlocalisations, rbbox_img = isess.run([image_4d, predictions, localisations, bbox_img],
                                                              feed_dict={img_input: img})

    # Get classes and bboxes from the net outputs.
    rclasses, rscores, rbboxes = np_methods.ssd_bboxes_select(
        rpredictions, rlocalisations, ssd_anchors,
        select_threshold=select_threshold, img_shape=net_shape, num_classes=2, decode=True)

    rbboxes = np_methods.bboxes_clip(rbbox_img, rbboxes)
    rclasses, rscores, rbboxes = np_methods.bboxes_sort(rclasses, rscores, rbboxes, top_k=400)
    rclasses, rscores, rbboxes = np_methods.bboxes_nms(rclasses, rscores, rbboxes, nms_threshold=nms_threshold)
    # Resize bboxes to original image shape. Note: useless for Resize.WARP!
    rbboxes = np_methods.bboxes_resize(rbbox_img, rbboxes)
    return rclasses, rscores, rbboxes


# Test on some demo image and visualize output.
path = '/home/yjin/data/demo/'
while True:
    image_name = input('Enter image name: ')
    image_path = Path(path + image_name)
    if image_path.is_file():
        img = mpimg.imread(image_path)
        rclasses, rscores, rbboxes = process_image(img, select_threshold=0.3, nms_threshold=0.1)
        visualization.plt_bboxes(img, rclasses, rscores, rbboxes)
        plt.show()
    elif image_name == 'exit':
        break
    else:
        print('File does not exist.')