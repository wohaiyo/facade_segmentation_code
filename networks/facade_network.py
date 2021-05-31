from __future__ import print_function
import tensorflow as tf
import tensorflow.contrib.slim as slim
import config as cfg
from libs.resnet import resnet_v1

def atrous_spp16(input_feature, depth=256):
    '''
    ASPP module for deeplabv3+
        if output_stride == 16, rates = [6, 12, 18];
        if output_stride == 8, rate:[12, 24, 36];

    :param input_feature: [b, h, w, c]
    '''
    with tf.variable_scope("aspp"):
        # 1x1 conv
        at_pooling1x1 = slim.conv2d(input_feature, depth, [1, 1], scope='conv1x1', activation_fn=None)

        # rate = 6
        at_pooling3x3_1 = slim.conv2d(input_feature, depth, [3, 3], scope='conv_3x3_1', rate=6, activation_fn=None)

        # rate = 12
        at_pooling3x3_2 = slim.conv2d(input_feature, depth, [3, 3], scope='conv_3x3_2', rate=12, activation_fn=None)

        # rate = 18
        at_pooling3x3_3 = slim.conv2d(input_feature, depth, [3, 3], scope='conv_3x3_3', rate=18, activation_fn=None)

        # image pooling
        img_pooling = tf.reduce_mean(input_feature, [1, 2], name='image_level_global_pooling', keep_dims=True)
        img_pooling = slim.conv2d(img_pooling, depth, [1, 1], scope='image_level_conv_1x1', activation_fn=None)
        img_pooling = tf.image.resize_bilinear(img_pooling, (input_feature.get_shape().as_list()[1],
                                                             input_feature.get_shape().as_list()[2]))

        net = tf.concat([img_pooling, at_pooling1x1, at_pooling3x3_1, at_pooling3x3_2, at_pooling3x3_3],
                        axis=3, name='atrous_concat')
        net = slim.conv2d(net, depth, [1, 1], scope='conv_1x1_output', activation_fn=None)

        return net

def inference_deeplabv3_plus_16(image):
    # Feature extractor: ResNet50
    with slim.arg_scope(resnet_v1.resnet_arg_scope()):
        net, end_points = resnet_v1.resnet_v1_50(image, num_classes=None, is_training=None,
                                                 global_pool=False, output_stride=16,
                                                 spatial_squeeze=False)
    # ASPP
    aspp = atrous_spp16(net)
    with tf.variable_scope('decoder'):
        # Low level
        low_level_features = end_points['resnet_v1_50/block1/unit_2/bottleneck_v1']
        low_level_features = slim.conv2d(low_level_features, 48, [1, 1], scope='low_level_feature_conv_1x1')
        low_level_features_shape = low_level_features.get_shape().as_list()[1:3]

        # Upsample
        net = tf.image.resize_images(aspp, low_level_features_shape)
        net = tf.concat([net, low_level_features], axis=3)
        net = slim.conv2d(net, 256, [3, 3], scope='conv_3x3_1')
        net = slim.conv2d(net, 256, [3, 3], scope='conv_3x3_2')

    # Classifier
    logits = slim.conv2d(net, cfg.DATASET_NUM_CLASSESS, [1, 1], scope='logits', activation_fn=None, normalizer_fn=None)
    img_shape = image.get_shape().as_list()
    logits = tf.image.resize_images(logits, [img_shape[1], img_shape[2]])

    label_pred = tf.expand_dims(tf.argmax(logits, axis=3, name="prediction"), dim=3)
    return label_pred, logits