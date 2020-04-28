from __future__ import print_function
import os,time,cv2, sys, math
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import time, datetime
import argparse
import random
import subprocess

# use 'Agg' on matplotlib so that plots could be generated even without Xserver
# running
import matplotlib
matplotlib.use('Agg')

from utils import utils, helpers
from builders import model_builder

import matplotlib.pyplot as plt

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="CamVid", help='Dataset you are using.')
parser.add_argument('--crop_height', type=int, default=512, help='Height of cropped input image to network')
parser.add_argument('--crop_width', type=int, default=512, help='Width of cropped input image to network')
parser.add_argument('--model', type=str, default="Regression", help='The model you are using. See model_builder.py for supported models')
parser.add_argument('--frontend', type=str, default="ResNet101", help='The frontend you are using. See frontend_builder.py for supported models')
parser.add_argument('--checkpoint_path', type=str, default="", help='The path to save the checkpoint')

args = parser.parse_args()

num_classes = 1

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess=tf.Session(config=config)

# Compute your softmax cross entropy loss
net_input = tf.placeholder(tf.float32,shape=[None,None,None,3])
net_output = tf.placeholder(tf.float32,shape=[None,None,None,num_classes])

network, init_fn = model_builder.build_model(model_name=args.model, frontend=args.frontend, net_input=net_input, num_classes=num_classes, crop_width=args.crop_width, crop_height=args.crop_height, is_training=True)

saver=tf.train.Saver(max_to_keep=1000)
sess.run(tf.global_variables_initializer())

utils.count_params()
if init_fn is not None:
    init_fn(sess)

saver.restore(sess, args.checkpoint_path)
# os.makedirs("Test2", exist_ok=True)

# Load the data
print("Loading the data ...")
train_input_names, train_output1_names, train_output2_names, val_input_names, val_output1_names, val_output2_names, test_input_names, test_output1_names, test_output2_names = utils.prepare_data_multi(dataset=args.dataset)

validation_loss_list = []
binary_loss_list = []

for ind in range(len(test_input_names)):
    input_image = np.expand_dims(np.float32(utils.load_image(test_input_names[ind])[:args.crop_height, :args.crop_width]),axis=0)/255.0
    
    # gt1 = utils.load_image(test_output1_names[ind])[:args.crop_height, :args.crop_width]
    # gt1 = helpers.reverse_one_hot(helpers.one_hot_it(gt1, label_values))

    gt2 = utils.load_image_output(test_output2_names[ind])[:args.crop_height, :args.crop_width]
    gt2 = np.float32(np.expand_dims(gt2, axis = -1))            
    # st = time.time()

    output1_image = sess.run(network, feed_dict={net_input:input_image})
    output1_image = np.array(output1_image[0,:,:,:])

    validation_loss, binary_loss = utils.evaluate_regression(pred=output1_image, label=gt2/255.0)

    validation_loss_list.append(validation_loss)
    binary_loss_list.append(binary_loss)
    
    # cv2.imwrite("%s/%s_org.png"%("Test2", file_name),cv2.cvtColor(np.uint8(input_image[0]*255.0), cv2.COLOR_RGB2BGR))
    # cv2.imwrite("%s/%s_pred1.png"%("Test2", file_name),cv2.cvtColor(np.uint8(out1_vis_image), cv2.COLOR_RGB2BGR))
    # cv2.imwrite("%s/%s_gt1.png"%("Test2", file_name),cv2.cvtColor(np.uint8(gt1), cv2.COLOR_RGB2BGR))

avg_validation_loss = np.mean(validation_loss_list)
avg_binary_loss = np.mean(binary_loss_list)

print("Test Regression Loss score = ", avg_validation_loss)
print("Test Binary Loss score =", avg_binary_loss)
