from __future__ import print_function
import os,time,cv2, sys, math
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import time, datetime
import argparse
import random
import subprocess
import pickle

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
parser.add_argument('--checkpoint_path', type=str, default="camvid_pspnet_multi", help='The path to save the checkpoint')
parser.add_argument('--model', type=str, default="PSPNetMultiTest", help='The path to save the checkpoint')
parser.add_argument('--frontend', type=str, default="ResNet101", help='The path to save the checkpoint')

args = parser.parse_args()
    
# Get the names of the classes so we can record the evaluation results
class_names_list, label_values = helpers.get_label_info(os.path.join(utils.dataset_dir[args.dataset]['path'], "class_dict.csv"))
class_names_string = ""
for class_name in class_names_list:
    if not class_name == class_names_list[-1]:
        class_names_string = class_names_string + class_name + ", "
    else:
        class_names_string = class_names_string + class_name

num_classes = len(label_values)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess=tf.Session(config=config)

# Compute your softmax cross entropy loss
net_input = tf.placeholder(tf.float32,shape=[None,None,None,3])
net_output1 = tf.placeholder(tf.float32,shape=[None,None,None,num_classes])
net_output2 = tf.placeholder(tf.float32,shape=[None,None,None,1])

network, init_fn = model_builder.build_model(model_name=args.model, frontend=args.frontend, net_input=net_input, num_classes=num_classes, crop_width=args.crop_width, crop_height=args.crop_height, is_training=True)

# network1, network2, intermediates = network[0], network[1], network[2]
intermediate1, intermediate2, intermediate3, net1, net2 = network[0], network[1], network[2], network[3], network[4]
#psp, seg, sal

saver=tf.train.Saver(max_to_keep=1000)
sess.run(tf.global_variables_initializer())

utils.count_params()

if init_fn is not None:
    init_fn(sess)

model_checkpoint_name = args.checkpoint_path+"/latest_model_" + "PSPNetMulti" + "_" + args.dataset + ".ckpt"
print('Loaded latest model checkpoint')
saver.restore(sess, model_checkpoint_name)

# Load the data
print("Loading the data ...")
train_input_names, train_output1_names, train_output2_names, val_input_names, val_output1_names, val_output2_names, test_input_names, test_output1_names, test_output2_names = utils.prepare_data_multi(dataset=args.dataset)

# Which validation images do we want
num_iters = 1
st = time.time()
epoch_st=time.time()

input_image_batch = []
input_test = range(0, 40)

# Collect a batch of images
for j in input_test:
    input_image = utils.load_image(train_input_names[j])
    with tf.device('/cpu:0'):
        x = random.randint(0, input_image.shape[1]-args.crop_width)
        y = random.randint(0, input_image.shape[0]-args.crop_height)
        input_image = input_image[y:y+args.crop_height, x:x+args.crop_width, :]
        # cv2.imwrite('Test/'+train_input_names[j].split('/')[-1], input_image)
        print(train_input_names[j].split('/')[-1])
        input_image = np.float32(input_image) / 255.0
        input_image_batch.append(np.expand_dims(input_image, axis = 0))

input_image_batch = np.squeeze(np.stack(input_image_batch, axis=1))  
intermediate2, intermediate3 = sess.run([intermediate2, intermediate3], feed_dict = {net_input:input_image_batch})

i = 0
# for segnet
# layer1, layer2, layer3, layer4 = intermediates
# for l1, l2, l3, l4 in zip(layer1, layer2, layer3, layer4):
#     x = [l1, l2, l3, l4]
#     pickle.dump(x, open(train_input_names[i].split('/')[-1][:-4]+'_outs.pkl', 'wb'))

# for pspnet
for out2, out3 in zip(intermediate2, intermediate3):
    name = train_input_names[input_test[i]].split('/')[-1][:-4]
    # pickle.dump(psp, open('Test/'+name+'_outs.pkl', 'wb'))
    pickle.dump(out2, open('Test/'+name+'_seg.pkl', 'wb'))
    pickle.dump(out3, open('Test/'+name+'_sal.pkl', 'wb'))
    # cv2.imwrite('Test/'+name+'_out2.png', out2*255.0)
    i += 1
