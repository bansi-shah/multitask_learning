import tensorflow as tf
from tensorflow.contrib import slim
import numpy as np
from builders import frontend_builder
import os, sys

def Upsampling(inputs,feature_map_shape):
    return tf.image.resize_bilinear(inputs, size=feature_map_shape)

def ConvUpscaleBlock(inputs, n_filters, kernel_size=[3, 3], scale=2):
    """
    Basic conv transpose block for Encoder-Decoder upsampling
    Apply successivly Transposed Convolution, BatchNormalization, ReLU nonlinearity
    """
    net = tf.nn.relu(slim.batch_norm(inputs, fused=True))
    net = slim.conv2d_transpose(net, n_filters, kernel_size=[3, 3], stride=[scale, scale], activation_fn=None)
    return net

def ConvBlock(inputs, n_filters, kernel_size=[3, 3]):
    """
    Basic conv block for Encoder-Decoder
    Apply successivly Convolution, BatchNormalization, ReLU nonlinearity
    """
    net = tf.nn.relu(slim.batch_norm(inputs, fused=True))
    net = slim.conv2d(net, n_filters, kernel_size, activation_fn=None, normalizer_fn=None)
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
    net = Upsampling(net, feature_map_shape)
    return net

def PyramidPoolingModule(inputs, feature_map_shape, pooling_type):
    """
    Build the Pyramid Pooling Module.
    """

    interp_block1 = InterpBlock(inputs, 1, feature_map_shape, pooling_type)
    interp_block2 = InterpBlock(inputs, 2, feature_map_shape, pooling_type)
    interp_block3 = InterpBlock(inputs, 3, feature_map_shape, pooling_type)
    interp_block6 = InterpBlock(inputs, 6, feature_map_shape, pooling_type)

    res = tf.concat([inputs, interp_block6, interp_block3, interp_block2, interp_block1], axis=-1)
    return res

def Attention(inputs, n_filters):
    # Global average pooling
    net = tf.reduce_mean(inputs, [1, 2], keep_dims=True)
    net = slim.conv2d(net, n_filters, kernel_size=[1, 1])
    net = slim.batch_norm(net, fused=True)
    #attention
    # net = tf.nn.softmax(net)
    net = tf.sigmoid(net)
    net = tf.multiply(inputs, net)
    return net

def build_pspnetx(inputs, label_size, num_classes, preset_model='PSPNetX', frontend="ResNet101", pooling_type = "MAX", weight_decay=1e-5, upscaling_method="conv", is_training=True, pretrained_dir="models"):
    """
    Builds the PSPNet model. 

    Arguments:
      inputs: The input tensor
      label_size: Size of the final label tensor. We need to know this for proper upscaling 
      preset_model: Which model you want to use. Select which ResNet model to use for feature extraction 
      num_classes: Number of classes
      pooling_type: Max or Average pooling

    Returns:
      PSPNet model
    """

    logits, end_points, frontend_scope, init_fn  = frontend_builder.build_frontend(inputs, frontend, pretrained_dir=pretrained_dir, is_training=is_training)

    feature_map_shape = [int(x / 8.0) for x in label_size]
    psp = PyramidPoolingModule(end_points['pool3'], feature_map_shape=feature_map_shape, pooling_type=pooling_type)

    #default upscaling is conv, exp4 bilinear, changes in function definition

    #simple task, dropout1 = 1.0, dropout2 = 0.9
    #difficult task, dropout1 = 0.9, dropout2 = 1.0


    """
    Net 1 will do segmentation, Net 2 will do Regression
    """

    #dropout1 = 0.4
    #dropout2 = 0.6

    #NEW ADDED LAYERS EXPERIMENT 1, 3
    # psp = slim.conv2d(psp, 512, [3, 3], activation_fn=None)
    # psp = slim.batch_norm(psp, fused=True)
    # psp = tf.nn.relu(psp)  
    # psp = slim.conv2d(psp, 512, [3, 3], activation_fn=None)
    # psp = slim.batch_norm(psp, fused=True)
    # psp = tf.nn.relu(psp)  
    # psp = slim.conv2d(psp, 512, [3, 3], activation_fn=None)
    # psp = slim.batch_norm(psp, fused=True)
    # psp = tf.nn.relu(psp)  
    ######DONE


    #NEW ADDED LAYERS EXPERIMENT 2, 3
    # net1 = slim.conv2d(psp, 1024, [3, 3], activation_fn=None)
    # net1 = slim.batch_norm(net1, fused=True)
    # net1 = tf.nn.relu(net1)
    ######DONE

    #NEW ADDED LAYERS EXPERIMENT DROPOUT
    #net1 = slim.dropout(psp, dropout1, is_training = is_training)
    ######DONE

    net1 = slim.conv2d(psp, 512, [3, 3], activation_fn=None)
    net1 = slim.batch_norm(net1, fused=True)
    net1 = tf.nn.relu(net1)

    if upscaling_method.lower() == "conv":
        net1 = ConvUpscaleBlock(net1, 256, kernel_size=[3, 3], scale=2)
        net1 = ConvBlock(net1, 256)
        net1 = ConvUpscaleBlock(net1, 128, kernel_size=[3, 3], scale=2)
        net1 = ConvBlock(net1, 128)
        net1 = ConvUpscaleBlock(net1, 64, kernel_size=[3, 3], scale=2)
        net1 = ConvBlock(net1, 64)
        
    elif upscaling_method.lower() == "bilinear":
        net1 = Upsampling(net1, label_size)
    
    net1 = slim.conv2d(net1, num_classes, [1, 1], activation_fn=None, scope='logits1')

    #NEW ADDED LAYERS EXPERIMENT 2,3
    # net2 = slim.conv2d(psp, 1024, [3, 3], activation_fn=None)
    # net2 = slim.batch_norm(net2, fused=True)
    # net2 = tf.nn.relu(net2)
    ######DONE    

    #NEW ADDED LAYERS EXPERIMENT DROPOUT
    #net2 = slim.dropout(psp, dropout2, is_training = is_training)
    ######DONE

    net2 = slim.conv2d(psp, 512, [3, 3], activation_fn=None)
    net2 = slim.batch_norm(net2, fused=True)
    net2 = tf.nn.relu(net2)

    if upscaling_method.lower() == "conv":
        net2 = ConvUpscaleBlock(net2, 256, kernel_size=[3, 3], scale=2)
        net2 = ConvBlock(net2, 256)
        net2 = ConvUpscaleBlock(net2, 128, kernel_size=[3, 3], scale=2)
        net2 = ConvBlock(net2, 128)
        net2 = ConvUpscaleBlock(net2, 64, kernel_size=[3, 3], scale=2)
        net2 = ConvBlock(net2, 64)

    elif upscaling_method.lower() == "bilinear":
        net2 = Upsampling(net2, label_size)

    #attention 32
    # net2 = ConvBlock(net2, 32)
    # net2 = Attention(net2, 64)
    
    # sigmoid / relu
    # net2 = slim.conv2d(net2, 1, [1, 1], activation_fn=tf.nn.sigmoid, scope='logits2')

    net2 = slim.conv2d(net2, 1, [1, 1], activation_fn=None, scope='logits2')
    return net1, net2, init_fn

def mean_image_subtraction(inputs, means=[123.68, 116.78, 103.94]):
    inputs=tf.to_float(inputs)
    num_channels = inputs.get_shape().as_list()[-1]
    if len(means) != num_channels:
      raise ValueError('len(means) must match the number of channels')
    channels = tf.split(axis=3, num_or_size_splits=num_channels, value=inputs)
    for i in range(num_channels):
        channels[i] -= means[i]
    return tf.concat(axis=3, values=channels)
