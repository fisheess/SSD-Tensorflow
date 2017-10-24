import math
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

import tf_extended as tfe
from nets import custom_layers
from nets import ssd_blocks
from nets import ssd_common


# =========================================================================== #
# SSD class definition.
# =========================================================================== #
class SSDNet(object):
    """Implementation of the modular SSD network with interchangable feature extractor
    """
    def __init__(self, feature_extractor, model_name):
        if model_name == 'ssd300':
            self.params = ssd_blocks.ssd300_params
        elif model_name == 'ssd512':
            self.params = ssd_blocks.ssd512_params
        else:
            raise ValueError('Model %s unknown. Choose either ssd300 or ssd512.' % model_name)
        self.feature_extractor = feature_extractor
        self.model_name = model_name
        self.params.feat_layers.insert(0, ssd_blocks.feat_layer[feature_extractor])

    # ======================================================================= #
    def net(self, inputs,
            is_training=True,
            update_feat_shapes=True,
            dropout_keep_prob=0.5,
            prediction_fn=slim.softmax,
            reuse=None):
        """Network definition.
        """
        r = _ssd_net(inputs, self.feature_extractor, self.model_name,
                     num_classes=self.params.num_classes,
                     feat_layers=self.params.feat_layers,
                     anchor_sizes=self.params.anchor_sizes,
                     anchor_ratios=self.params.anchor_ratios,
                     normalizations=self.params.normalizations,
                     is_training=is_training,
                     dropout_keep_prob=dropout_keep_prob,
                     prediction_fn=prediction_fn,
                     reuse=reuse)
        # Update feature shapes (try at least!)
        if update_feat_shapes:
            self.params.feat_shapes = \
                _ssd_feat_shapes_from_net(r[0], self.params.feat_shapes)
        return r

    def arg_scope(self, weight_decay=0.0005, data_format='NHWC'):
        """Network arg_scope.
        """
        return ssd_blocks.ssd_arg_scope(weight_decay, data_format=data_format)

    # ======================================================================= #
    def anchors(self, img_shape, dtype=np.float32):
        """Compute the default anchor boxes, given an image shape.
        """
        return _ssd_anchors_all_layers(img_shape,
                                       self.params.feat_shapes,
                                       self.params.anchor_sizes,
                                       self.params.anchor_ratios,
                                       self.params.anchor_steps,
                                       self.params.anchor_offset,
                                       dtype)

    def bboxes_encode(self, labels, bboxes, anchors,
                      scope=None):
        """Encode labels and bounding boxes.
        """
        return ssd_common.tf_ssd_bboxes_encode(
            labels, bboxes, anchors,
            self.params.num_classes,
            self.params.no_annotation_label,
            ignore_threshold=0.5,
            prior_scaling=self.params.prior_scaling,
            scope=scope)

    def bboxes_decode(self, feat_localizations, anchors,
                      scope='ssd_bboxes_decode'):
        """Encode labels and bounding boxes.
        """
        return ssd_common.tf_ssd_bboxes_decode(
            feat_localizations, anchors,
            prior_scaling=self.params.prior_scaling,
            scope=scope)

    def detected_bboxes(self, predictions, localisations,
                        select_threshold=None, nms_threshold=0.5,
                        clipping_bbox=None, top_k=400, keep_top_k=200):
        """Get the detected bounding boxes from the SSD network output.
        """
        # Select top_k bboxes from predictions, and clip
        rscores, rbboxes = \
            ssd_common.tf_ssd_bboxes_select(predictions, localisations,
                                            select_threshold=select_threshold,
                                            num_classes=self.params.num_classes)
        rscores, rbboxes = \
            tfe.bboxes_sort(rscores, rbboxes, top_k=top_k)
        # Apply NMS algorithm.
        rscores, rbboxes = \
            tfe.bboxes_nms_batch(rscores, rbboxes,
                                 nms_threshold=nms_threshold,
                                 keep_top_k=keep_top_k)
        # if clipping_bbox is not None:
        #     rbboxes = tfe.bboxes_clip(clipping_bbox, rbboxes)
        return rscores, rbboxes

    def losses(self, logits, localisations,
               gclasses, glocalisations, gscores,
               match_threshold=0.5,
               negative_ratio=3.,
               alpha=1.,
               label_smoothing=0.,
               scope='ssd_losses'):
        """Define the SSD network losses.
        """
        return _ssd_losses(logits, localisations,
                           gclasses, glocalisations, gscores,
                           match_threshold=match_threshold,
                           negative_ratio=negative_ratio,
                           alpha=alpha,
                           label_smoothing=label_smoothing,
                           scope=scope)


# =========================================================================== #
# Functional definition of VGG-based SSD 300.
# =========================================================================== #
def tensor_shape(x, rank=3):
    """Returns the dimensions of a tensor.
    Args:
      x: A N-D Tensor of shape.
    Returns:
      A list of dimensions. Dimensions that are statically known are python
        integers,otherwise they are integer scalar tensors.
    """
    if x.get_shape().is_fully_defined():
        return x.get_shape().as_list()
    else:
        static_shape = x.get_shape().with_rank(rank).as_list()
        dynamic_shape = tf.unstack(tf.shape(x), rank)
        return [s if s is not None else d
                for s, d in zip(static_shape, dynamic_shape)]


def _ssd_multibox_layer(inputs,
                        num_classes,
                        sizes,
                        ratios=[1],
                        normalization=-1,
                        bn_normalization=False):
    """Construct a multibox layer, return a class and localization predictions.
    """
    net = inputs
    if normalization > 0:
        net = custom_layers.l2_normalization(net, scaling=True)
    # Number of anchors.
    num_anchors = len(sizes) + len(ratios)

    # Location.
    num_loc_pred = num_anchors * 4
    loc_pred = slim.conv2d(net, num_loc_pred, [3, 3], activation_fn=None,
                           scope='conv_loc')
    loc_pred = custom_layers.channel_to_last(loc_pred)
    loc_pred = tf.reshape(loc_pred,
                          tensor_shape(loc_pred, 4)[:-1] + [num_anchors, 4])
    # Class prediction.
    num_cls_pred = num_anchors * num_classes
    cls_pred = slim.conv2d(net, num_cls_pred, [3, 3], activation_fn=None,
                           scope='conv_cls')
    cls_pred = custom_layers.channel_to_last(cls_pred)
    cls_pred = tf.reshape(cls_pred,
                          tensor_shape(cls_pred, 4)[:-1] + [num_anchors, num_classes])
    return cls_pred, loc_pred


def _ssd_net(inputs, feature_extractor, model_name,
             num_classes, feat_layers, anchor_sizes, anchor_ratios, normalizations,
             is_training=True, dropout_keep_prob=0.5, prediction_fn=slim.softmax, reuse=None):
    _feature_extractor = ssd_blocks.get_base_network_fn(feature_extractor)
    if model_name == 'ssd300':
        _ssd_blocks = ssd_blocks.ssd300_blocks
    else:
        _ssd_blocks = ssd_blocks.ssd512_blocks
    scope = feature_extractor + '_' + model_name
    with tf.variable_scope(scope, 'ssd', [inputs], reuse=reuse):
        net, end_points = _feature_extractor(inputs)
        with ssd_blocks.ssd_arg_scope():
            with tf.variable_scope(model_name):
                net, end_points = _ssd_blocks(net, end_points)
        predictions = []
        logits = []
        localisations = []
        for i, layer in enumerate(feat_layers):
            with tf.variable_scope(layer + '_box'):
                pre, loc = _ssd_multibox_layer(end_points[layer],
                                               num_classes,
                                               anchor_sizes[i],
                                               anchor_ratios[i],
                                               normalizations[i])
            predictions.append(prediction_fn(pre))
            logits.append(pre)
            localisations.append(loc)

        return predictions, localisations, logits, end_points


# =========================================================================== #
# SSD tools...
# =========================================================================== #
def ssd_size_bounds_to_values(size_bounds,
                              n_feat_layers,
                              model_name):
    """Compute the reference sizes of the anchor boxes from relative bounds.
    The absolute values are measured in pixels, based on the network
    default size.

    This function follows the computation performed in the original
    implementation of SSD in Caffe.

    Return:
      list of list containing the absolute sizes at each scale. For each scale,
      the ratios only apply to the first value.
    """
    if model_name == 'ssd300':
        img_shape = (300, 300)
    elif model_name == 'ssd512':
        img_shape = (512, 512)
    else:
        raise ValueError('Model name %s is unknown.' % model_name)

    img_size = img_shape[0]
    min_ratio = int(size_bounds[0] * 100)
    max_ratio = int(size_bounds[1] * 100)
    step = int(math.floor((max_ratio - min_ratio) / (n_feat_layers - 2)))
    # Start with the following smallest sizes.
    if model_name == 'ssd300':
        sizes = [[img_size * size_bounds[0] / 2, img_size * size_bounds[0]]]
    else:
        sizes = [[img_size * 0.04, img_size * 0.1]]
    for ratio in range(min_ratio, max_ratio + 1, step):
        sizes.append([img_size * ratio / 100.,
                      img_size * (ratio + step) / 100.])
    return sizes


def _ssd_feat_shapes_from_net(predictions, default_shapes=None):
    """Try to obtain the feature shapes from the prediction layers. The latter
    can be either a Tensor or Numpy ndarray.

    Return:
      list of feature shapes. Default values if predictions shape not fully
      determined.
    """
    feat_shapes = []
    for l in predictions:
        # Get the shape, from either a np array or a tensor.
        if isinstance(l, np.ndarray):
            shape = l.shape
        else:
            shape = l.get_shape().as_list()
        shape = shape[1:4]
        # Problem: undetermined shape...
        if None in shape:
            return default_shapes
        else:
            feat_shapes.append(shape)
    return feat_shapes


def _ssd_anchor_one_layer(img_shape,
                          feat_shape,
                          sizes,
                          ratios,
                          step,
                          offset=0.5,
                          dtype=np.float32):
    """Computer SSD default anchor boxes for one feature layer.

    Determine the relative position grid of the centers, and the relative
    width and height.

    Arguments:
      feat_shape: Feature shape, used for computing relative position grids;
      sizes: Absolute reference sizes;
      ratios: Ratios to use on these features;
      img_shape: Image shape, used for computing height, width relatively to the
        former;
      offset: Grid offset.

    Return:
      y, x, h, w: Relative x and y grids, and height and width.
    """
    # Compute the position grid: simple way.
    # y, x = np.mgrid[0:feat_shape[0], 0:feat_shape[1]]
    # y = (y.astype(dtype) + offset) / feat_shape[0]
    # x = (x.astype(dtype) + offset) / feat_shape[1]
    # Weird SSD-Caffe computation using steps values...
    y, x = np.mgrid[0:feat_shape[0], 0:feat_shape[1]]
    y = (y.astype(dtype) + offset) * step / img_shape[0]
    x = (x.astype(dtype) + offset) * step / img_shape[1]

    # Expand dims to support easy broadcasting.
    y = np.expand_dims(y, axis=-1)
    x = np.expand_dims(x, axis=-1)

    # Compute relative height and width.
    # Tries to follow the original implementation of SSD for the order.
    num_anchors = len(sizes) + len(ratios)
    h = np.zeros((num_anchors,), dtype=dtype)
    w = np.zeros((num_anchors,), dtype=dtype)
    # Add first anchor boxes with ratio=1.
    h[0] = sizes[0] / img_shape[0]
    w[0] = sizes[0] / img_shape[1]
    di = 1
    if len(sizes) > 1:
        h[1] = math.sqrt(sizes[0] * sizes[1]) / img_shape[0]
        w[1] = math.sqrt(sizes[0] * sizes[1]) / img_shape[1]
        di += 1
    for i, r in enumerate(ratios):
        h[i + di] = sizes[0] / img_shape[0] / math.sqrt(r)
        w[i + di] = sizes[0] / img_shape[1] * math.sqrt(r)
    return y, x, h, w


def _ssd_anchors_all_layers(img_shape,
                            layers_shape,
                            anchor_sizes,
                            anchor_ratios,
                            anchor_steps,
                            offset=0.5,
                            dtype=np.float32):
    """Compute anchor boxes for all feature layers.
    """
    layers_anchors = []
    for i, s in enumerate(layers_shape):
        anchor_bboxes = _ssd_anchor_one_layer(img_shape, s,
                                              anchor_sizes[i],
                                              anchor_ratios[i],
                                              anchor_steps[i],
                                              offset=offset, dtype=dtype)
        layers_anchors.append(anchor_bboxes)
    return layers_anchors


# =========================================================================== #
# SSD loss function.
# =========================================================================== #
def _ssd_losses(logits, localisations,
                gclasses, glocalisations, gscores,
                match_threshold=0.5,
                negative_ratio=3.,
                alpha=1.,
                label_smoothing=0.,
                device='/cpu:0',
                scope=None):
    with tf.name_scope(scope, 'ssd_losses'):
        lshape = tfe.get_shape(logits[0], 5)
        num_classes = lshape[-1]
        batch_size = lshape[0]

        # Flatten out all vectors!
        flogits = []
        fgclasses = []
        fgscores = []
        flocalisations = []
        fglocalisations = []
        for i in range(len(logits)):
            flogits.append(tf.reshape(logits[i], [-1, num_classes]))
            fgclasses.append(tf.reshape(gclasses[i], [-1]))
            fgscores.append(tf.reshape(gscores[i], [-1]))
            flocalisations.append(tf.reshape(localisations[i], [-1, 4]))
            fglocalisations.append(tf.reshape(glocalisations[i], [-1, 4]))
        # And concat the crap!
        logits = tf.concat(flogits, axis=0)
        gclasses = tf.concat(fgclasses, axis=0)
        gscores = tf.concat(fgscores, axis=0)
        localisations = tf.concat(flocalisations, axis=0)
        glocalisations = tf.concat(fglocalisations, axis=0)
        dtype = logits.dtype

        # Compute positive matching mask...
        pmask = gscores > match_threshold
        fpmask = tf.cast(pmask, dtype)
        n_positives = tf.reduce_sum(fpmask)

        # Hard negative mining...
        no_classes = tf.cast(pmask, tf.int32)
        predictions = slim.softmax(logits)
        nmask = tf.logical_and(tf.logical_not(pmask),
                               gscores > -0.5)
        fnmask = tf.cast(nmask, dtype)
        nvalues = tf.where(nmask,
                           predictions[:, 0],
                           1. - fnmask)
        nvalues_flat = tf.reshape(nvalues, [-1])
        # Number of negative entries to select.
        max_neg_entries = tf.cast(tf.reduce_sum(fnmask), tf.int32)
        n_neg = tf.cast(negative_ratio * n_positives, tf.int32) + batch_size
        n_neg = tf.minimum(n_neg, max_neg_entries)

        val, idxes = tf.nn.top_k(-nvalues_flat, k=n_neg)
        max_hard_pred = -val[-1]
        # Final negative mask.
        nmask = tf.logical_and(nmask, nvalues < max_hard_pred)
        fnmask = tf.cast(nmask, dtype)

        # Add cross-entropy loss.
        with tf.name_scope('cross_entropy_pos'):
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                  labels=gclasses)
            loss = tf.div(tf.reduce_sum(loss * fpmask), batch_size, name='value')
            tf.losses.add_loss(loss)

        with tf.name_scope('cross_entropy_neg'):
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                  labels=no_classes)
            loss = tf.div(tf.reduce_sum(loss * fnmask), batch_size, name='value')
            tf.losses.add_loss(loss)

        # Add localization loss: smooth L1, L2, ...
        with tf.name_scope('localization'):
            # Weights Tensor: positive mask + random negative.
            weights = tf.expand_dims(alpha * fpmask, axis=-1)
            loss = custom_layers.abs_smooth(localisations - glocalisations)
            loss = tf.div(tf.reduce_sum(loss * weights), batch_size, name='value')
            tf.losses.add_loss(loss)
