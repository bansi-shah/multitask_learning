from __future__ import division
import os,time,cv2
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

def conv_block(inputs, n_filters, kernel_size=[3, 3], dropout_p=0.0):
	"""
	Basic conv block for Encoder-Decoder
	Apply successivly Convolution, BatchNormalization, ReLU nonlinearity
	Dropout (if dropout_p > 0) on the inputs
	"""
	conv = slim.conv2d(inputs, n_filters, kernel_size, activation_fn=None, normalizer_fn=None)
	out = tf.nn.relu(slim.batch_norm(conv, fused=True))
	if dropout_p != 0.0:
	  out = slim.dropout(out, keep_prob=(1.0-dropout_p))
	return out

def conv_transpose_block(inputs, n_filters, kernel_size=[3, 3], dropout_p=0.0):
	"""
	Basic conv transpose block for Encoder-Decoder upsampling
	Apply successivly Transposed Convolution, BatchNormalization, ReLU nonlinearity
	Dropout (if dropout_p > 0) on the inputs
	"""
	conv = slim.conv2d_transpose(inputs, n_filters, kernel_size=[3, 3], stride=[2, 2], activation_fn=None)
	out = tf.nn.relu(slim.batch_norm(conv))
	if dropout_p != 0.0:
	  out = slim.dropout(out, keep_prob=(1.0-dropout_p))
	return out

def build_segnet_multi(inputs, num_classes, preset_model = "Encoder-Decoder-Multi", dropout_p=0.5, scope=None):
	"""
	Builds the Encoder-Decoder-Multitask model. Inspired by SegNet with some modifications
	Optionally includes skip connections

	Arguments:
	  inputs: the input tensor
	  n_classes: number of classes
	  dropout_p: dropout rate applied after each convolution (0. for not using)

	Returns:
	  Encoder-Decoder model
	"""
	has_skip = True

	#####################
	# Downsampling path #
	#####################
	net = conv_block(inputs, 64)
	net = conv_block(net, 64)
	net = slim.pool(net, [2, 2], stride=[2, 2], pooling_type='MAX')
	skip_1 = net

	net = conv_block(net, 128)
	net = conv_block(net, 128)
	net = slim.pool(net, [2, 2], stride=[2, 2], pooling_type='MAX')
	skip_2 = net

	net = conv_block(net, 256)
	net = conv_block(net, 256)
	net = conv_block(net, 256)
	net = slim.pool(net, [2, 2], stride=[2, 2], pooling_type='MAX')
	skip_3 = net

	net = conv_block(net, 512)
	net = conv_block(net, 512)
	net = conv_block(net, 512)
	net = slim.pool(net, [2, 2], stride=[2, 2], pooling_type='MAX')
	skip_4 = net

	net = conv_block(net, 512)
	net = conv_block(net, 512)
	net = conv_block(net, 512)
	net = slim.pool(net, [2, 2], stride=[2, 2], pooling_type='MAX')


	#####################
	# Upsampling path #
	#####################
	net1 = conv_transpose_block(net, 512)
	net1 = conv_block(net1, 512)
	net1 = conv_block(net1, 512)
	net1 = conv_block(net1, 512)
	if has_skip:
		net1 = tf.add(net1, skip_4)

	net1 = conv_transpose_block(net1, 512)
	net1 = conv_block(net1, 512)
	net1 = conv_block(net1, 512)
	net1 = conv_block(net1, 256)
	if has_skip:
		net1 = tf.add(net1, skip_3)

	net1 = conv_transpose_block(net1, 256)
	net1 = conv_block(net1, 256)
	net1 = conv_block(net1, 256)
	net1 = conv_block(net1, 128)
	if has_skip:
		net1 = tf.add(net1, skip_2)

	net1 = conv_transpose_block(net1, 128)
	net1 = conv_block(net1, 128)
	net1 = conv_block(net1, 64)
	if has_skip:
		net1 = tf.add(net1, skip_1)

	net1 = conv_transpose_block(net1, 64)
	net1 = conv_block(net1, 64)
	net1 = conv_block(net1, 64)

	#####################
	#      Softmax      #
	#####################
	net1 = slim.conv2d(net1, num_classes, [1, 1], activation_fn=tf.nn.relu, scope='logits1')

	#####################
	# Upsampling path #
	#####################
	net2 = conv_transpose_block(net, 512)
	net2 = conv_block(net2, 512)
	net2 = conv_block(net2, 512)
	net2 = conv_block(net2, 512)
	if has_skip:
		net2 = tf.add(net2, skip_4)

	net2 = conv_transpose_block(net2, 512)
	net2 = conv_block(net2, 512)
	net2 = conv_block(net2, 512)
	net2 = conv_block(net2, 256)
	if has_skip:
		net2 = tf.add(net2, skip_3)

	net2 = conv_transpose_block(net2, 256)
	net2 = conv_block(net2, 256)
	net2 = conv_block(net2, 256)
	net2 = conv_block(net2, 128)
	if has_skip:
		net2 = tf.add(net2, skip_2)

	net2 = conv_transpose_block(net2, 128)
	net2 = conv_block(net2, 128)
	net2 = conv_block(net2, 64)
	if has_skip:
		net2 = tf.add(net2, skip_1)

	net2 = conv_transpose_block(net2, 64)
	net2 = conv_block(net2, 64)
	net2 = conv_block(net2, 64)

	#####################
	#      Softmax      #
	#####################
	net2 = slim.conv2d(net1, 1, [1, 1], activation_fn=tf.nn.relu, scope='logits2')

	return net1, net2