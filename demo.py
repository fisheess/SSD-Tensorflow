from os import path
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
from pims import ImageSequence
import time

from nets import modular_ssd, np_methods, ssd_vgg_512
from preprocessing import ssd_vgg_preprocessing
from notebooks.visualization import plt_bboxes

class_names = {
    1: 'aeroplane',
    2: 'bicycle',
    3: 'bird',
    4: 'boat',
    5: 'bottle',
    6: 'bus',
    7: 'car',
    8: 'cat',
    9: 'chair',
    10: 'cow',
    11: 'diningtable',
    12: 'dog',
    13: 'horse',
    14: 'motorbike',
    15: 'person',
    16: 'pottedplant',
    17: 'sheep',
    18: 'sofa',
    19: 'train',
    20: 'tvmonitor'
}

demo_params = ssd_vgg_512.SSDParams(
    img_shape=(512, 512),
    num_classes=21,
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
    anchor_offset=1.5,
    normalizations=[20, -1, -1, -1, -1, -1, -1],
    prior_scaling=[0.1, 0.1, 0.2, 0.2]
)

ckpt_file = '/home/yjin/SSD/experiments/ssd_512_vgg_finetune_04-12-2017/model.ckpt-150000'


# TensorFlow session: grow memory when needed.
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1, allow_growth=True)
config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
isess = tf.InteractiveSession(config=config)

# Input placeholder.
net_shape = demo_params.img_shape
data_format = 'NHWC'
img_input = tf.placeholder(tf.uint8, shape=(None, None, 3))
# Evaluation pre-processing: resize to SSD net shape.
image_pre, labels_pre, bboxes_pre, bbox_img = ssd_vgg_preprocessing.preprocess_for_eval(
    img_input, None, None, net_shape, data_format, resize=ssd_vgg_preprocessing.Resize.WARP_RESIZE)
image_4d = tf.expand_dims(image_pre, 0)

# Define the SSD model.
ssd_net = ssd_vgg_512.SSDNet(demo_params)
predictions, localisations, _, _ = ssd_net.net(image_4d, is_training=False)
# Restore SSD model.
isess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(isess, ckpt_file)
# SSD default anchor boxes.
ssd_anchors = ssd_net.anchors(net_shape)
print('Model loaded successfully.')


# Main image processing routine.
def process_image(img, select_threshold=0.5, nms_threshold=.45, net_shape=(512, 512)):
    # Run SSD network.
    rimg, rpredictions, rlocalisations, rbbox_img = isess.run([image_4d, predictions, localisations, bbox_img],
                                                              feed_dict={img_input: img})

    # Get classes and bboxes from the net outputs.
    rclasses, rscores, rbboxes = np_methods.ssd_bboxes_select(
        rpredictions, rlocalisations, ssd_anchors,
        select_threshold=select_threshold, img_shape=net_shape, num_classes=21, decode=True)

    rbboxes = np_methods.bboxes_clip(rbbox_img, rbboxes)
    rclasses, rscores, rbboxes = np_methods.bboxes_sort(rclasses, rscores, rbboxes, top_k=400)
    rclasses, rscores, rbboxes = np_methods.bboxes_nms(rclasses, rscores, rbboxes, nms_threshold=nms_threshold)
    # Resize bboxes to original image shape. Note: useless for Resize.WARP!
    rbboxes = np_methods.bboxes_resize(rbbox_img, rbboxes)
    return rclasses, rscores, rbboxes


def convert_to_abs_bboxes(rbboxes, image_shape):
    """
    convert relative bboxes to absolute bboxes according to the requirement of module RectDraw
    :param rbboxes: [ymin, xmin, ymax, xmax], all in range(0,1)
    :param image_shape: [height, width, channels]
    :return: bbox: numpy array [xmin, ymin, width, height], absolute values
    """
    bboxes = []
    for rbbox in rbboxes:
        xmin = int(rbbox[1] * image_shape[1])
        ymin = int(rbbox[0] * image_shape[0])
        width = int((rbbox[3] - rbbox[1]) * image_shape[1])
        height = int((rbbox[2] - rbbox[0]) * image_shape[0])
        bboxes.append([xmin, ymin, width, height])
    return bboxes


def plot2array(fig):
    imarray = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    imarray = imarray.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return imarray


# Test on some demo image and visualize output.
def visualize_single_img(img):
    time0 = time.time()
    rclasses, rscores, rbboxes = process_image(img, select_threshold=0.5, nms_threshold=0.45)
    elapsed = int((time.time() - time0) * 1000)
    print('{:d} ms | {:d} detection(s)'.format(elapsed, len(rclasses)))
    plt_bboxes(img, rclasses, rscores, rbboxes)


def visualize_img_seq(vid_dir='/home/yjin/data/vid', img_format='jpg', debug=False, ):
    vid = ImageSequence(path.join(vid_dir, '*.' + img_format))
    height = vid[0].shape[0]
    width = vid[0].shape[1]
    fig = plt.figure(figsize=(10, 10))
    ax = plt.subplot()
    plt.show(block=False)
    # imout = []
    while True:
        time0 = time.time()
        for frame in vid:
            #if debug:
            #    time1 = time.time()
            #    print('%d ms for loading image.' % int((time1 - time0) * 1000))
            rclasses, rscores, rbboxes = process_image(frame, select_threshold=0.5, nms_threshold=0.45)
            #if debug:
            #    time2 = time.time()
            #    print('%d ms for detection.' % int((time2 - time1) * 1000))
            ax.clear()
            plt.imshow(frame)
            for i in range(rclasses.shape[0]):
                cls_id = int(rclasses[i])
                if cls_id >= 0:
                    ymin = int(rbboxes[i, 0] * height)
                    xmin = int(rbboxes[i, 1] * width)
                    ymax = int(rbboxes[i, 2] * height)
                    xmax = int(rbboxes[i, 3] * width)
                    rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                         fill=False, edgecolor='green', linewidth=1.5)
                    ax.add_patch(rect)
                    if debug:
                        plt.text(xmin, ymin - 2, '{:s} | {:.3f}'.format(class_names[cls_id], rscores[i]),
                                 bbox=dict(facecolor='green', alpha=0.5), fontsize=8, color='white')
            fps = 1. / (time.time() - time0)
            plt.text(0, 0, '{:.2f} fps | {:d} detection(s)'.format(fps, len(rbboxes)),
                     bbox=dict(facecolor='green', alpha=0.5), fontsize=12, color='white')
            #if debug:
            #    time3 = time.time()
            #    print('%d ms for refreshing image.' % int((time3 - time2) * 1000))
            time0 = time.time()
            # fig.canvas.draw()
            # im = plot2array(fig)
            # imout.append(im)
            plt.pause(0.0001)
        if input('Play again? (y/any key except y): ') != 'y':
            break
    # return imout


if __name__ == '__main__':
    while True:
        choice = input('chose image(i), video(v) or quit(q): ')
        if choice == 'i' or choice == 'image':
            image_dir = '/home/yjin/data/demo'
            while True:
                image_name = input('Enter image filename or quit(q): ')
                image_path = path.join(image_dir, image_name)
                if path.isfile(image_path):
                    img = mpimg.imread(image_path)
                    visualize_single_img(img)
                elif image_name == 'q' or image_name == 'quit':
                    break
                else:
                    print('File does not exist.')
        elif choice == 'v' or choice == 'video':
            vid_dir = input('Directory to video: ')
            img_format = input('Image format: ')
            # imout =
            visualize_img_seq(vid_dir, img_format, debug=False)
            # vidwriter = cv2.VideoWriter('/home/yjin/data/test.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 8, (1000, 1000))
            # for im in imout:
            #     vidwriter.write(im[..., ::-1])
        elif choice == 'q' or choice == 'quit':
            break
        else:
            print("I don't understand what '{}' means.".format(choice))
