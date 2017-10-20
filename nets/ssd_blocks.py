from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
from collections import namedtuple
import tensorflow as tf
import tensorflow.contrib.slim as slim

from nets import vgg
from nets import mobilenet_v1
from nets import custom_layers


# =========================================================================== #
# Definition of the default parameters
# =========================================================================== #
SSDParams = namedtuple('SSDParameters', ['model_name',
                                         'img_shape',
                                         'num_classes',
                                         'no_annotation_label',
                                         'feature_layers',
                                         'feature_shapes',
                                         'anchor_size_bounds',
                                         'anchor_sizes',
                                         'anchor_ratios',
                                         'anchor_steps',
                                         'anchor_offset',
                                         'normalizations',
                                         'prior_scaling'])

ssd300_params = SSDParams(model_name='ssd300',
                          img_shape=(300, 300),
                          num_classes=21,
                          no_annotation_label=21,
                          feature_layers=['ssd_block7', 'ssd_block8', 'ssd_block9', 'ssd_block10', 'ssd_block11'],
                          feature_shapes=[(38, 38), (19, 19), (10, 10), (5, 5), (3, 3), (1, 1)],
                          anchor_size_bounds=[0.15, 0.90],
                          anchor_sizes=[(21., 45.),
                                        (45., 99.),
                                        (99., 153.),
                                        (153., 207.),
                                        (207., 261.),
                                        (261., 315.)],
                          anchor_ratios=[[2, .5],
                                         [2, .5, 3, 1. / 3],
                                         [2, .5, 3, 1. / 3],
                                         [2, .5, 3, 1. / 3],
                                         [2, .5],
                                         [2, .5]],
                          anchor_steps=[8, 16, 32, 64, 100, 300],
                          anchor_offset=0.5,
                          normalizations=[20, -1, -1, -1, -1, -1],
                          prior_scaling=[0.1, 0.1, 0.2, 0.2]
                          )

ssd512_params = SSDParams(model_name='ssd512',
                          img_shape=(512, 512),
                          num_classes=21,
                          no_annotation_label=21,
                          feature_layers=['ssd_block7', 'ssd_block8', 'ssd_block9', 'ssd_block10', 'ssd_block11', 'ssd_block12'],
                          feature_shapes=[(64, 64), (32, 32), (16, 16), (8, 8), (4, 4), (2, 2), (1, 1)],
                          anchor_size_bounds=[0.10, 0.90],
                          anchor_sizes=[(20.48, 51.2),
                                        (51.2, 133.12),
                                        (133.12, 215.04),
                                        (215.04, 296.96),
                                        (296.96, 378.88),
                                        (378.88, 460.8),
                                        (460.8, 542.72)],
                          anchor_ratios=[[2, .5],
                                         [2, .5, 3, 1. / 3],
                                         [2, .5, 3, 1. / 3],
                                         [2, .5, 3, 1. / 3],
                                         [2, .5, 3, 1. / 3],
                                         [2, .5],
                                         [2, .5]],
                          anchor_steps=[8, 16, 32, 64, 128, 256, 512],
                          anchor_offset=0.5,
                          normalizations=[20, -1, -1, -1, -1, -1, -1],
                          prior_scaling=[0.1, 0.1, 0.2, 0.2]
                          )


# =========================================================================== #
# Implementation of the SSD blocks
# =========================================================================== #
def ssd300_blocks(net, end_points):
    # block 6: 3x3 conv
    net = slim.conv2d(net, 1024, [3, 3], rate=6, scope='conv6')
    net = slim.batch_norm(net)
    end_points['ssd_block6'] = net
    # block 7: 1x1 conv
    net = slim.conv2d(net, 1024, [1, 1], scope='conv7')
    net = slim.batch_norm(net)
    end_points['ssd_block7'] = net
    # block 8/9/10/11: 1x1 and 3x3 convolutions with stride 2 (except lasts)
    end_point = 'ssd_block8'
    with tf.variable_scope(end_point):
        net = slim.conv2d(net, 256, [1, 1], scope='conv1x1')
        net = slim.batch_norm(net)
        net = custom_layers.pad2d(net, pad=(1, 1))
        net = slim.conv2d(net, 512, [3, 3], stride=2, scope='conv3x3', padding='VALID')
        net = slim.batch_norm(net)
    end_points[end_point] = net
    end_point = 'ssd_block9'
    with tf.variable_scope(end_point):
        net = slim.conv2d(net, 128, [1, 1], scope='conv1x1')
        net = slim.batch_norm(net)
        net = custom_layers.pad2d(net, pad=(1, 1))
        net = slim.conv2d(net, 256, [3, 3], stride=2, scope='conv3x3', padding='VALID')
        net = slim.batch_norm(net)
    end_points[end_point] = net
    end_point = 'ssd_block10'
    with tf.variable_scope(end_point):
        net = slim.conv2d(net, 128, [1, 1], scope='conv1x1')
        net = slim.batch_norm(net)
        net = slim.conv2d(net, 256, [3, 3], scope='conv3x3', padding='VALID')
        net = slim.batch_norm(net)
    end_points[end_point] = net
    end_point = 'ssd_block11'
    with tf.variable_scope(end_point):
        net = slim.conv2d(net, 128, [1, 1], scope='conv1x1')
        net = slim.batch_norm(net)
        net = slim.conv2d(net, 256, [3, 3], scope='conv3x3', padding='VALID')
        net = slim.batch_norm(net)
    end_points[end_point] = net
    return net, end_points


def ssd512_blocks(net, end_points):
    # Block 6: 3x3 conv
    net = slim.conv2d(net, 1024, [3, 3], rate=6, scope='conv6')
    net = slim.batch_norm(net)
    end_points['ssd_block6'] = net
    # Block 7: 1x1 conv
    net = slim.conv2d(net, 1024, [1, 1], scope='conv7')
    net = slim.batch_norm(net)
    end_points['ssd_block7'] = net
    # Block 8/9/10/11/12: 1x1 and 3x3 convolutions stride 2 (except last).
    end_point = 'ssd_block8'
    with tf.variable_scope(end_point):
        net = slim.conv2d(net, 256, [1, 1], scope='conv1x1')
        net = slim.batch_norm(net)
        net = custom_layers.pad2d(net, pad=(1, 1))
        net = slim.conv2d(net, 512, [3, 3], stride=2, scope='conv3x3', padding='VALID')
        net = slim.batch_norm(net)
    end_points[end_point] = net
    end_point = 'ssd_block9'
    with tf.variable_scope(end_point):
        net = slim.conv2d(net, 128, [1, 1], scope='conv1x1')
        net = slim.batch_norm(net)
        net = custom_layers.pad2d(net, pad=(1, 1))
        net = slim.conv2d(net, 256, [3, 3], stride=2, scope='conv3x3', padding='VALID')
        net = slim.batch_norm(net)
    end_points[end_point] = net
    end_point = 'ssd_block10'
    with tf.variable_scope(end_point):
        net = slim.conv2d(net, 128, [1, 1], scope='conv1x1')
        net = slim.batch_norm(net)
        net = custom_layers.pad2d(net, pad=(1, 1))
        net = slim.conv2d(net, 256, [3, 3], stride=2, scope='conv3x3', padding='VALID')
        net = slim.batch_norm(net)
    end_points[end_point] = net
    end_point = 'ssd_block11'
    with tf.variable_scope(end_point):
        net = slim.conv2d(net, 128, [1, 1], scope='conv1x1')
        net = slim.batch_norm(net)
        net = custom_layers.pad2d(net, pad=(1, 1))
        net = slim.conv2d(net, 256, [3, 3], stride=2, scope='conv3x3', padding='VALID')
        net = slim.batch_norm(net)
    end_points[end_point] = net
    end_point = 'ssd_block12'
    with tf.variable_scope(end_point):
        net = slim.conv2d(net, 128, [1, 1], scope='conv1x1')
        net = slim.batch_norm(net)
        net = custom_layers.pad2d(net, pad=(1, 1))
        net = slim.conv2d(net, 256, [4, 4], scope='conv4x4', padding='VALID')
        net = slim.batch_norm(net)
    end_points[end_point] = net
    return net, end_points


# =========================================================================== #
# Mapping base networks, their arg_scops and corresponding feature layer
# =========================================================================== #
feature_layer = {'alexnet_v2': '',
                 'cifarnet': '',
                 'overfeat': '',
                 'vgg_a': 'vgg_a/conv4/conv4_3',
                 'vgg_16': 'vgg_16/conv4/conv4_3',
                 'vgg_19': 'vgg_19/conv4/conv4_3',
                 'inception_v1': '',
                 'inception_v2': '',
                 'inception_v3': '',
                 'inception_v4': '',
                 'inception_resnet_v2': '',
                 'lenet': '',
                 'resnet_v1_50': '',
                 'resnet_v1_101': '',
                 'resnet_v1_152': '',
                 'resnet_v1_200': '',
                 'resnet_v2_50': '',
                 'resnet_v2_101': '',
                 'resnet_v2_152': '',
                 'resnet_v2_200': '',
                 'mobilenet_v1': 'Conv2d_11_pointwise',
                 'mobilenet_v1_075': 'Conv2d_11_pointwise',
                 'mobilenet_v1_050': 'Conv2d_11_pointwise',
                 'mobilenet_v1_025': 'Conv2d_11_pointwise',
                 'xception': ''
                 }


base_networks_map = {'mobilenet_v1': mobilenet_v1.mobilenet_v1_base_ssd,
                     'vgg_a': vgg.vgg_a_base,
                     'vgg_16': vgg.vgg_16_base,
                     'vgg_19': vgg.vgg_19_base
                     }


base_arg_scopes_map = {'vgg_16': vgg.vgg_base_arg_scope,
                       'mobilenet_v1': mobilenet_v1_base_arg_scope
                       }


def get_base_network_fn(name):
    """Returns a base_network_fn such as 'net, end_points = base_network_fn(images)'.

    Args:
      name: The name of the network.
      num_classes: The number of classes to use for classification.
      weight_decay: The l2 coefficient for the model weights.
      is_training: `True` if the model is being used for training and `False`
        otherwise.

    Returns:
      base_network_fn: A function that applies the model to a batch of images. It has
        the following signature:
          nets, end_points = network_fn(images)
    Raises:
      ValueError: If network `name` is not recognized.
    """
    if name not in base_networks_map:
        raise ValueError('Name of network unknown: %s' % name)
    func = base_networks_map[name]

    @functools.wraps(func)
    def base_network_fn(images):
        arg_scope = base_arg_scopes_map[name]()
        # arg_scope = arg_scopes_map[name](weight_decay=weight_decay)
        with slim.arg_scope(arg_scope):
            return func(images)

    if hasattr(func, 'default_image_size'):
        base_network_fn.default_image_size = func.default_image_size

    return base_network_fn


