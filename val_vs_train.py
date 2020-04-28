from __future__ import print_function
import os,time,cv2, sys, math
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import time, datetime
import argparse
import random
import subprocess
# use 'Agg' on matplotlib so that plots could be generated even without Xserver running
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
parser.add_argument('--num_epochs', type=int, default=300, help='Number of epochs to train for')
parser.add_argument('--epoch_start_i', type=int, default=0, help='Start counting epochs from this number')
parser.add_argument('--checkpoint_step', type=int, default=5, help='How often to save checkpoints (epochs)')
parser.add_argument('--validation_step', type=int, default=1, help='How often to perform validation (epochs)')
parser.add_argument('--image', type=str, default=None, help='The image you want to predict on. Only valid in "predict" mode.')
parser.add_argument('--continue_training', type=str2bool, default=False, help='Whether to continue training from a checkpoint')
parser.add_argument('--dataset', type=str, default="CamVid", help='Dataset you are using.')
parser.add_argument('--crop_height', type=int, default=512, help='Height of cropped input image to network')
parser.add_argument('--crop_width', type=int, default=512, help='Width of cropped input image to network')
parser.add_argument('--batch_size', type=int, default=8, help='Number of images in each batch')
parser.add_argument('--num_val_images', type=int, default=20, help='The number of images to used for validations')
parser.add_argument('--h_flip', type=str2bool, default=False, help='Whether to randomly flip the image horizontally for data augmentation')
parser.add_argument('--v_flip', type=str2bool, default=False, help='Whether to randomly flip the image vertically for data augmentation')
parser.add_argument('--brightness', type=float, default=None, help='Whether to randomly change the image brightness for data augmentation. Specifies the max bightness change as a factor between 0.0 and 1.0. For example, 0.1 represents a max brightness change of 10%% (+-).')
parser.add_argument('--rotation', type=float, default=None, help='Whether to randomly rotate the image for data augmentation. Specifies the max rotation angle in degrees.')
parser.add_argument('--model', type=str, default="Regression", help='The model you are using. See model_builder.py for supported models')
parser.add_argument('--frontend', type=str, default="ResNet101", help='The frontend you are using. See frontend_builder.py for supported models')
parser.add_argument('--checkpoint_path', type=str, default="checkpoints_3", help='The path to save the checkpoint')
args = parser.parse_args()

def data_augmentation(input_image, output_image):
    input_image, output_image = utils.random_crop(input_image, output_image, args.crop_height, args.crop_width)
    if args.h_flip and random.randint(0,1):
        input_image = cv2.flip(input_image, 1)
        output_image = cv2.flip(output_image, 1)
    if args.v_flip and random.randint(0,1):
        input_image = cv2.flip(input_image, 0)
        output_image = cv2.flip(output_image, 0)
    if args.brightness:
        factor = 1.0 + random.uniform(-1.0*args.brightness, args.brightness)
        table = np.array([((i / 255.0) * factor) * 255 for i in np.arange(0, 256)]).astype(np.uint8)
        input_image = cv2.LUT(input_image, table)
    if args.rotation:
        angle = random.uniform(-1*args.rotation, args.rotation)
    if args.rotation:
        M = cv2.getRotationMatrix2D((input_image.shape[1]//2, input_image.shape[0]//2), angle, 1.0)
        input_image = cv2.warpAffine(input_image, M, (input_image.shape[1], input_image.shape[0]), flags=cv2.INTER_NEAREST)
        output_image = cv2.warpAffine(output_image, M, (output_image.shape[1], output_image.shape[0]), flags=cv2.INTER_NEAREST)

    return input_image, output_image

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
num_classes = 1

net_input = tf.placeholder(tf.float32,shape = [None, None, None, 3])
net_output = tf.placeholder(tf.float32,shape = [None, None, None, num_classes])

network, init_fn = model_builder.build_model(model_name = args.model, frontend = args.frontend, net_input =net_input, num_classes = num_classes, crop_width = args.crop_width, crop_height = args.crop_height, is_training = True)

loss = tf.reduce_mean(tf.losses.mean_squared_error(labels = net_output, predictions = network))

saver = tf.train.Saver(max_to_keep=1000)
sess.run(tf.global_variables_initializer())

utils.count_params()

if init_fn is not None:
    init_fn(sess)

print("Loading the data ...")
train_input_names, train_output_names, val_input_names, val_output_names, test_input_names, test_output_names = utils.prepare_data(dataset_dir=args.dataset)

print("Dataset -->", args.dataset)
print("Model -->", args.model)
print("Crop Height -->", args.crop_height)
print("Crop Width -->", args.crop_width)
print("Num Epochs -->", args.num_epochs)
print("Batch Size -->", args.batch_size)

print("Data Augmentation:")
print("\tVertical Flip -->", args.v_flip)
print("\tHorizontal Flip -->", args.h_flip)
print("\tBrightness Alteration -->", args.brightness)
print("\tRotation -->", args.rotation)
print()

random.seed(16)

current_val_losses = []
current_train_losses = []
current_test_losses = []

print("Data collected")

for model in range(0, 200, 5):
    model_checkpoint_name = "%s/%04d/model.ckpt"%(args.checkpoint_path, model)
    saver.restore(sess, model_checkpoint_name)

    # current_losses = []
    # num_iters = int(np.floor(len(train_output_names) / args.batch_size))

    # for i in range(num_iters):
    #     input_image_batch = []
    #     output_image_batch = []

    #     for j in range(args.batch_size):
    #         index = i*args.batch_size + j
    #         input_image = utils.load_image(train_input_names[index])
    #         output_image = utils.load_image_output(train_output_names[index])
            
    #         with tf.device('/cpu:0'):
    #             input_image, output_image = data_augmentation(input_image, output_image)

    #             # Prep the data. Make sure the labels are in one-hot format
    #             input_image = np.float32(input_image) / 255.0
    #             # output_image = np.float32(helpers.one_hot_it(label=output_image, label_values=label_values))
    #             output_image = np.float32(np.expand_dims(output_image, axis = -1)) / 255.0

    #             input_image_batch.append(np.expand_dims(input_image, axis = 0))
    #             output_image_batch.append(np.expand_dims(output_image, axis = 0))

    #     if args.batch_size == 1:
    #         input_image_batch = input_image_batch[0]
    #         output_image_batch = output_image_batch[0]
    #     else:
    #         input_image_batch = np.squeeze(np.stack(input_image_batch, axis=1))
    #         output_image_batch = np.squeeze(np.stack(output_image_batch, axis=1))
    #         output_image_batch = np.expand_dims(output_image_batch, axis = -1)

    #     current = sess.run(loss ,feed_dict={net_input:input_image_batch,net_output:output_image_batch})
    #     current_losses.append(current)
    #     print(current)

    # current_train_losses.append(np.mean(current_losses))

    # current_losses = []
    # num_iters = int(np.floor(len(val_output_names) / args.batch_size))

    # for i in range(num_iters):
    #     input_image_batch = []
    #     output_image_batch = []

    #     for j in range(args.batch_size):
    #         index = i*args.batch_size + j
    #         input_image = utils.load_image(val_input_names[index])
    #         output_image = utils.load_image_output(val_output_names[index])
            
    #         with tf.device('/cpu:0'):
    #             input_image, output_image = data_augmentation(input_image, output_image)

    #             # Prep the data. Make sure the labels are in one-hot format
    #             input_image = np.float32(input_image) / 255.0
    #             # output_image = np.float32(helpers.one_hot_it(label=output_image, label_values=label_values))
    #             output_image = np.float32(np.expand_dims(output_image, axis = -1)) / 255.0

    #             input_image_batch.append(np.expand_dims(input_image, axis = 0))
    #             output_image_batch.append(np.expand_dims(output_image, axis = 0))

    #     if args.batch_size == 1:
    #         input_image_batch = input_image_batch[0]
    #         output_image_batch = output_image_batch[0]
    #     else:
    #         input_image_batch = np.squeeze(np.stack(input_image_batch, axis=1))
    #         output_image_batch = np.squeeze(np.stack(output_image_batch, axis=1))
    #         output_image_batch = np.expand_dims(output_image_batch, axis = -1)

    #     current = sess.run(loss ,feed_dict={net_input:input_image_batch,net_output:output_image_batch})
    #     current_losses.append(current)
    #     print(current)

    # current_val_losses.append(np.mean(current_losses))

    current_losses = []
    num_iters = int(np.floor(len(test_output_names) / args.batch_size))

    for i in range(num_iters):
        input_image_batch = []
        output_image_batch = []

        for j in range(args.batch_size):
            index = i*args.batch_size + j
            input_image = utils.load_image(test_input_names[index])
            output_image = utils.load_image_output(test_output_names[index])
            
            with tf.device('/cpu:0'):
                input_image, output_image = data_augmentation(input_image, output_image)

                # Prep the data. Make sure the labels are in one-hot format
                input_image = np.float32(input_image) / 255.0
                # output_image = np.float32(helpers.one_hot_it(label=output_image, label_values=label_values))
                output_image = np.float32(np.expand_dims(output_image, axis = -1)) / 255.0

                input_image_batch.append(np.expand_dims(input_image, axis = 0))
                output_image_batch.append(np.expand_dims(output_image, axis = 0))

        if args.batch_size == 1:
            input_image_batch = input_image_batch[0]
            output_image_batch = output_image_batch[0]
        else:
            input_image_batch = np.squeeze(np.stack(input_image_batch, axis=1))
            output_image_batch = np.squeeze(np.stack(output_image_batch, axis=1))
            output_image_batch = np.expand_dims(output_image_batch, axis = -1)

        current = sess.run(loss ,feed_dict={net_input:input_image_batch,net_output:output_image_batch})
        current_losses.append(current)

    current_test_losses.append(np.mean(current_losses))

# fig1, ax1 = plt.subplots(figsize=(11, 8))
# ax1.plot(range(len(current_train_losses)), current_train_losses)
# ax1.set_title("Train loss vs epochs")
# ax1.set_xlabel("Epoch")
# ax1.set_ylabel("Current loss")
# plt.savefig(args.checkpoint_path +'/train_loss_vs_epochs.png')
# plt.clf()

# fig2, ax2 = plt.subplots(figsize=(11, 8))
# ax2.plot(range(len(current_val_losses)), current_val_losses)
# ax2.set_title("Val loss vs epochs")
# ax2.set_xlabel("Epoch")
# ax2.set_ylabel("Current loss")
# plt.savefig(args.checkpoint_path +'/val_loss_vs_epochs.png')
# plt.clf()

fig3, ax3 = plt.subplots(figsize=(11, 8))
ax3.plot(range(len(current_test_losses)), current_test_losses)
ax3.set_title("Test loss vs epochs")
ax3.set_xlabel("Epoch")
ax3.set_ylabel("Current loss")
plt.savefig(args.checkpoint_path +'/test_loss_vs_epochs.png')
plt.clf()
