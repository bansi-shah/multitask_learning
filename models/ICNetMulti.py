import tensorflow as tf
from tensorflow.contrib import slim
import numpy as np
from builders import frontend_builder
import os, sys
from frontends import resnet_v2

def Upsampling_by_shape(inputs, feature_map_shape):
    return tf.image.resize_bilinear(inputs, size=feature_map_shape)

def Upsampling_by_scale(inputs, scale):
    return tf.image.resize_bilinear(inputs, size=[tf.shape(inputs)[1]*scale,  tf.shape(inputs)[2]*scale])

def ConvUpscaleBlock(inputs, n_filters, kernel_size=[3, 3], scale=2):
    """
    Basic conv transpose block for Encoder-Decoder upsampling
    Apply successivly Transposed Convolution, BatchNormalization, ReLU nonlinearity
    """
    net = tf.nn.relu(slim.batch_norm(net, fused=True))
    net = slim.conv2d_transpose(inputs, n_filters, kernel_size=[3, 3], stride=[2, 2], activation_fn=None)
    return net

def ConvBlock(inputs, n_filters, kernel_size=[3, 3]):
    """
    Basic conv block for Encoder-Decoder
    Apply successivly Convolution, BatchNormalization, ReLU nonlinearity
    """
    net = slim.conv2d(inputs, n_filters, kernel_size, activation_fn=None, normalizer_fn=None)
    net = tf.nn.relu(slim.batch_norm(net, fused=True))
    return net

def InterpBlock(net, level, feature_map_shape, pooling_type):
    # Compute the kernel and stride sizes according to how large the final feature map will be
    # When the kernel size and strides are equal, then we can compute the final feature map size
    # by simply dividing the current size by the kernel or stride size
    # The final feature map sizes are 1x1, 2x2, 3x3, and 6x6. We round to the closest integer
    kernel_size = [int(np.round(float(feature_map_shape[0]) / float(level))), int(np.round(float(feature_map_shape[1]) / float(level)))]
    stride_size = kernel_size

    net = slim.pool(net, kernel_size, stride=stride_size, pooling_type='MAX')
    net = slim.conv2d(net, 512, [1, 1], activation_fn=None)
    net = slim.batch_norm(net, fused=True)
    net = tf.nn.relu(net)
    net = Upsampling_by_shape(net, feature_map_shape)
    return net

def PyramidPoolingModule(inputs, feature_map_shape, pooling_type):
    """
    Build the Pyramid Pooling Module.
    """

    interp_block1 = InterpBlock(inputs, 1, feature_map_shape, pooling_type)
    interp_block2 = InterpBlock(inputs, 2, feature_map_shape, pooling_type)
    interp_block3 = InterpBlock(inputs, 3, feature_map_shape, pooling_type)
    interp_block6 = InterpBlock(inputs, 6, feature_map_shape, pooling_type)
    inputs = Upsampling_by_shape(inputs, feature_map_shape)

    res = tf.concat([inputs, interp_block6, interp_block3, interp_block2, interp_block1], axis = -1)
    return res

def CFFBlock(F1, F2, num_classes):
    F1_big = Upsampling_by_scale(F1, scale=2)
    F1_out = slim.conv2d(F1_big, num_classes, [1, 1], activation_fn=None)

    F1_big = slim.conv2d(F1_big, 512, [3, 3], rate=2, activation_fn=None)
    F1_big = slim.batch_norm(F1_big, fused=True)

    F2_proj = slim.conv2d(F2, 512, [1, 1], rate=1, activation_fn=None)
    F2_proj = slim.batch_norm(F2_proj, fused=True)

    F1_big = tf.image.resize_bilinear(F1_big, tf.shape(F2_proj)[1:3])

    F2_out = tf.add(F1_big, F2_proj)
    F2_out = tf.nn.relu(F2_out)

    return F1_out, F2_out

def build_icnet_multi(inputs, label_size, num_classes, preset_model='ICNet', pooling_type = "MAX",
    frontend="ResNet101", weight_decay=1e-5, is_training=True, pretrained_dir="models"):
    """
    Builds the ICNet model. 

    Arguments:
      inputs: The input tensor
      label_size: Size of the final label tensor. We need to know this for proper upscaling 
      preset_model: Which model you want to use. Select which ResNet model to use for feature extraction 
      num_classes: Number of classes
      pooling_type: Max or Average pooling

    Returns:
      ICNet model
    """

    inputs_4 = Upsampling_by_scale(inputs, 4)   
    inputs_2 = Upsampling_by_scale(inputs, 2)
    inputs_1 = inputs

    if frontend == 'ResNet50':
        with slim.arg_scope(resnet_v2.resnet_arg_scope(weight_decay=weight_decay)):
            logits_32, end_points_32 = resnet_v2.resnet_v2_50(inputs_4, is_training=is_training, scope='resnet_v2_50')
            logits_16, end_points_16 = resnet_v2.resnet_v2_50(inputs_2, is_training=is_training, scope='resnet_v2_50')
            logits_8, end_points_8 = resnet_v2.resnet_v2_50(inputs_1, is_training=is_training, scope='resnet_v2_50')
            resnet_scope='resnet_v2_50'
            # ICNet requires pre-trained ResNet weights
            init_fn = slim.assign_from_checkpoint_fn(os.path.join(pretrained_dir, 'resnet_v2_50.ckpt'), slim.get_model_variables('resnet_v2_50'), ignore_missing_vars=True)
    elif frontend == 'ResNet101':
        with slim.arg_scope(resnet_v2.resnet_arg_scope(weight_decay=weight_decay)):
            logits_32, end_points_32 = resnet_v2.resnet_v2_101(inputs_4, is_training=is_training, scope='resnet_v2_101')
            logits_16, end_points_16 = resnet_v2.resnet_v2_101(inputs_2, is_training=is_training, scope='resnet_v2_101', reuse = True)
            logits_8, end_points_8 = resnet_v2.resnet_v2_101(inputs_1, is_training=is_training, scope='resnet_v2_101', reuse = True)
            # ICNet requires pre-trained ResNet weights
            init_fn = slim.assign_from_checkpoint_fn(os.path.join(pretrained_dir, 'resnet_v2_101.ckpt'), slim.get_model_variables('resnet_v2_101'), ignore_missing_vars=True)
    elif frontend == 'ResNet152':
        with slim.arg_scope(resnet_v2.resnet_arg_scope(weight_decay=weight_decay)):
            logits_32, end_points_32 = resnet_v2.resnet_v2_152(inputs_4, is_training=is_training, scope='resnet_v2_152')
            logits_16, end_points_16 = resnet_v2.resnet_v2_152(inputs_2, is_training=is_training, scope='resnet_v2_152')
            logits_8, end_points_8 = resnet_v2.resnet_v2_152(inputs_1, is_training=is_training, scope='resnet_v2_152')
            resnet_scope='resnet_v2_152'
            # ICNet requires pre-trained ResNet weights
            init_fn = slim.assign_from_checkpoint_fn(os.path.join(pretrained_dir, 'resnet_v2_152.ckpt'), slim.get_model_variables('resnet_v2_152')) 
    else:
        raise ValueError("Unsupported ResNet model '%s'. This function only supports ResNet 50, ResNet 101, and ResNet 152" % (frontend))

    feature_map_shape = [int(x / 32.0) for x in label_size]

    psp_32 = PyramidPoolingModule(end_points_32['pool3'], feature_map_shape = feature_map_shape, pooling_type = pooling_type)
    
    out_16_1, block_16_1 = CFFBlock(psp_32, end_points_16['pool3'], num_classes)
    out_8_1, block_8_1 = CFFBlock(block_16_1, end_points_8['pool3'], num_classes)
    out_4_1 = Upsampling_by_scale(out_8_1, scale = 2)
    net_1 = slim.conv2d(out_4_1, num_classes, [1, 1], activation_fn = None, scope = 'logits1') 
    # out_final = Upsampling_by_scale(out_4, scale = 2)
    # out_full = tf.concat([out_16, out_8, out_4, out_final], axis = -1)
    # net = slim.conv2d(out_final, num_classes, [1, 1], activation_fn=None, scope='logits')

    out_16_2, block_16_2 = CFFBlock(psp_32, end_points_16['pool3'], num_classes)
    out_8_2, block_8_2 = CFFBlock(block_16_2, end_points_8['pool3'], num_classes)
    out_4_2 = Upsampling_by_scale(out_8_2, scale = 2)
    net_2 = slim.conv2d(out_4_2, 1, [1, 1], activation_fn = tf.nn.sigmoid, scope = 'logits2') 
    # out_final = Upsampling_by_scale(out_4, scale = 2)
    # out_full = tf.concat([out_16, out_8, out_4, out_final], axis = -1)
    # net = slim.conv2d(out_final, num_classes, [1, 1], activation_fn=None, scope='logits'
    return net_1, net_2, init_fn

def mean_image_subtraction(inputs, means=[123.68, 116.78, 103.94]):
    inputs=tf.to_float(inputs)
    num_channels = inputs.get_shape().as_list()[-1]
    if len(means) != num_channels:
      raise ValueError('len(means) must match the number of channels')
    channels = tf.split(axis=3, num_or_size_splits=num_channels, value=inputs)
    for i in range(num_channels):
        channels[i] -= means[i]
    return tf.concat(axis=3, values=channels)