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

from utils import utils, helpers, conf
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
parser.add_argument('--num_epochs', type=int, default= 300, help='Number of epochs to train for')
parser.add_argument('--epoch_start_i', type=int, default=0, help='Start counting epochs from this number')
parser.add_argument('--checkpoint_step', type=int, default=200, help='How often to save checkpoints (epochs)')
parser.add_argument('--validation_step', type=int, default=2, help='How often to perform validation (epochs)')
parser.add_argument('--image', type=str, default=None, help='The image you want to predict on. Only valid in "predict" mode.')
parser.add_argument('--continue_training', type=str2bool, default=False, help='Whether to continue training from a checkpoint')
parser.add_argument('--dataset', type=str, default="Osie", help='Dataset you are using.')
parser.add_argument('--crop_height', type=int, default=512, help='Height of cropped input image to network')
parser.add_argument('--crop_width', type=int, default=512, help='Width of cropped input image to network')
parser.add_argument('--batch_size', type=int, default=8, help='Number of images in each batch')
parser.add_argument('--num_val_images', type=int, default=20, help='The number of images to used for validations')
parser.add_argument('--h_flip', type=str2bool, default=False, help='Whether to randomly flip the image horizontally for data augmentation')
parser.add_argument('--v_flip', type=str2bool, default=False, help='Whether to randomly flip the image vertically for data augmentation')
parser.add_argument('--brightness', type=float, default=None, help='Whether to randomly change the image brightness for data augmentation. Specifies the max bightness change as a factor between 0.0 and 1.0. For example, 0.1 represents a max brightness change of 10%% (+-).')
parser.add_argument('--rotation', type=float, default=None, help='Whether to randomly rotate the image for data augmentation. Specifies the max rotation angle in degrees.')
parser.add_argument('--model', type=str, default="PSPNet", help='The model you are using. See model_builder.py for supported models')
parser.add_argument('--frontend', type=str, default="ResNet101", help='The frontend you are using. See frontend_builder.py for supported models')
parser.add_argument('--checkpoint_path', type=str, default="osie_pspnet", help='The path to save the checkpoint')
parser.add_argument('--resize', type=str, default=True, help='Crop or resize image')

args = parser.parse_args()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
num_classes = 1
prev_loss = 100000000000

net_input = tf.placeholder(tf.float32,shape=[None,None,None,3])
net_output = tf.placeholder(tf.float32,shape=[None,None,None,num_classes])

network, init_fn = model_builder.build_model(model_name=args.model, frontend=args.frontend, net_input=net_input, num_classes=num_classes, crop_width=args.crop_width, crop_height=args.crop_height, is_training=True)

loss = tf.reduce_mean(tf.losses.mean_squared_error(labels = net_output, predictions = network))

# opt = tf.train.RMSPropOptimizer(learning_rate=0.0001, decay=0.995).minimize(loss, var_list=[var for var in tf.trainable_variables()])
opt = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss, var_list=[var for var in tf.trainable_variables()])

saver = tf.train.Saver(max_to_keep=1000)
sess.run(tf.global_variables_initializer())

utils.count_params()

if init_fn is not None:
    init_fn(sess)

# Load a previous checkpoint if desired
model_checkpoint_name = args.checkpoint_path+"/latest_model_" + args.model + "_" + args.dataset + ".ckpt"
if args.continue_training:
    print('Loaded latest model checkpoint')
    saver.restore(sess, model_checkpoint_name)

# Load the data
print("Loading the data ...")
train_input_names, train_output_names, val_input_names, val_output_names, test_input_names, test_output_names = utils.prepare_data(dataset=args.dataset)

print("\n***** Begin training *****")
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

avg_loss_per_epoch = []
avg_mse_loss_per_epoch = []
avg_mae_loss_per_epoch = []
avg_pre_per_epoch = []
avg_recall_per_epoch = []
avg_f1_per_epoch = []

# Which validation images do we want
val_indices = []
num_vals = min(args.num_val_images, len(val_input_names))

# Set random seed to make sure models are validated on the same validation images.
# So you can compare the results of different models more intuitively.
random.seed(16)
val_indices=random.sample(range(0,len(val_input_names)),num_vals)

# Do the training here
for epoch in range(args.epoch_start_i, args.num_epochs):
    current_losses = []
    cnt=0

    # Equivalent to shuffling
    id_list = np.random.permutation(len(train_input_names))
    num_iters = int(np.floor(len(id_list) / args.batch_size))
    st = time.time()
    epoch_st=time.time()
    for i in range(num_iters):
        # st=time.time()

        input_image_batch = []
        output_image_batch = []

        # Collect a batch of images
        for j in range(args.batch_size):
            index = i*args.batch_size + j
            id = id_list[index]
            # print(train_input_names[id])
            input_image = utils.load_image(train_input_names[id])
            output_image = utils.load_image_output(train_output_names[id])

            # print(input_image.shape, output_image.shape)

            with tf.device('/cpu:0'):
                input_image, output_image = utils.data_augmentation(input_image, output_image, args)

                input_image = np.float32(input_image) / 255.0
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

        # Do the training
        _, current = sess.run([opt, loss],feed_dict={net_input:input_image_batch,net_output:output_image_batch})
        current_losses.append(current)

        cnt = cnt + args.batch_size
        if cnt % 20 == 0:
            string_print = "Epoch = %d Count = %d Current_Loss = %.4f Time = %.2f"%(epoch,cnt,current,time.time()-st)
            utils.LOG(string_print)
            st = time.time()

    # Create directories if needed
    # if not os.path.isdir("%s/%04d"%(args.checkpoint_path,epoch)):
    #     os.makedirs("%s/%04d"%(args.checkpoint_path,epoch))

    # Save latest checkpoint to same file name

    print("Saving latest checkpoint")
    saver.save(sess, model_checkpoint_name)

    if epoch % args.validation_step == 0:
        print("Performing validation")
        mse_loss_list = []
        mae_loss_list = []
        precision_list = []
        recall_list = []
        f1_list = []

        for ind in val_indices:
            input_image = utils.load_image(val_input_names[ind])
            output_image = utils.load_image_output(val_output_names[ind])
            input_image, gt = utils.data_augmentation(input_image, output_image, args)
            input_image = np.expand_dims(np.float32(input_image),axis=0)/255.0
            gt = np.float32(np.expand_dims(gt, axis = -1))
            # st = time.time()

            output_image = sess.run(network,feed_dict={net_input:input_image})
            mse, mae = utils.evaluate_regression(pred=output_image, label=gt/255.0)
            # mse, mae, precision, recall, f1 = utils.evaluate_regression(pred=output_image, label=gt/255.0)
            # mse_loss_list.append(mse)
            # mae_loss_list.append(mae)
            # precision_list.append(precision)
            # recall_list.append(recall)
            # f1_list.append(f1)
                
            if prev_loss > mse:
                prev_loss = mse
                saver.save(sess,"%s/best_model.ckpt"%(args.checkpoint_path))

            # output_image = np.array(output_image[0,:,:,:])*255
        
            # file_name = utils.filepath_to_name(val_input_names[ind])
            # file_name = os.path.basename(val_input_names[ind])
            # file_name = os.path.splitext(file_name)[0]
            # cv2.imwrite("%s/%04d/%s_pred.png"%(args.checkpoint_path, epoch, file_name),cv2.cvtColor(np.uint8(output_image), cv2.COLOR_RGB2BGR))
            # cv2.imwrite("%s/%04d/%s_gt.png"%(args.checkpoint_path, epoch, file_name),cv2.cvtColor(np.uint8(gt), cv2.COLOR_RGB2BGR))
            # cv2.imwrite("%s/%04d/%s_org.png"%(args.checkpoint_path, epoch, file_name),cv2.cvtColor(np.uint8(img), cv2.COLOR_RGB2BGR))

    epoch_time=time.time()-epoch_st
    remain_time=epoch_time*(args.num_epochs-1-epoch)
    m, s = divmod(remain_time, 60)
    h, m = divmod(m, 60)
    if s!=0:
        train_time="Remaining training time = %d hours %d minutes %d seconds\n"%(h,m,s)
    else:
        train_time="Remaining training time = 0, Training completed.\n"
    utils.LOG(train_time)
    
    mean_loss = np.mean(current_losses)
    avg_loss_per_epoch.append(mean_loss)
    
    avg_mse_loss = np.mean(mse_loss_list)
    avg_mse_loss_per_epoch.append(avg_mse_loss)

    avg_mae_loss = np.mean(mae_loss_list)
    avg_mae_loss_per_epoch.append(avg_mae_loss)
    
    # mean_precision = np.mean(precision_list)
    # avg_pre_per_epoch.append(mean_precision)
    # mean_recall = np.mean(recall_list)
    # avg_recall_per_epoch.append(mean_recall)
    # mean_f1 = np.mean(f1_list)
    # avg_f1_per_epoch.append(mean_f1)

    print('Validation mean sqaured error :', avg_mse_loss)
    print('Validation mean absolute error :', avg_mae_loss)
    # print('Validation mean precision :', mean_precision)
    # print('Validation mean recall :', mean_recall)
    # print('Validation mean f1:', mean_f1)

    fig1, ax1 = plt.subplots(figsize=(11, 8))
    ax1.plot(range(epoch+1), avg_loss_per_epoch)
    ax1.set_title("Train loss vs epochs")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Current loss")
    plt.savefig(args.checkpoint_path +'/train_loss.png')
    plt.clf()

    fig1, ax1 = plt.subplots(figsize=(11, 8))
    ax1.plot(range(args.epoch_start_i, epoch+1), avg_mse_loss_per_epoch)
    ax1.set_title("Validation mse loss vs epochs")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Current loss")
    plt.savefig(args.checkpoint_path+ '/val_mse_loss.png')
    plt.clf()

    fig1, ax1 = plt.subplots(figsize=(11, 8))
    ax1.plot(range(args.epoch_start_i, epoch+1), avg_mae_loss_per_epoch)
    ax1.set_title("Validation mae loss vs epochs")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Current loss")
    plt.savefig(args.checkpoint_path+ '/val_mae_loss.png')
    plt.clf()

    # fig1, ax1 = plt.subplots(figsize=(11, 8))
    # ax1.plot(range(args.epoch_start_i, epoch+1), avg_pre_per_epoch)
    # ax1.set_title("Validation precision vs epochs")
    # ax1.set_xlabel("Epoch")
    # ax1.set_ylabel("Current loss")
    # plt.savefig(args.checkpoint_path+ '/val_precision.png')
    # plt.clf()

    # fig1, ax1 = plt.subplots(figsize=(11, 8))
    # ax1.plot(range(args.epoch_start_i, epoch+1), avg_recall_per_epoch)
    # ax1.set_title("Validation recall vs epochs")
    # ax1.set_xlabel("Epoch")
    # ax1.set_ylabel("Current loss")
    # plt.savefig(args.checkpoint_path+ '/val_recall.png')
    # plt.clf()

    # fig1, ax1 = plt.subplots(figsize=(11, 8))
    # ax1.plot(range(args.epoch_start_i, epoch+1), avg_f1_per_epoch)
    # ax1.set_title("Validation f1 vs epochs")
    # ax1.set_xlabel("Epoch")
    # ax1.set_ylabel("Current loss")
    # plt.savefig(args.checkpoint_path+ '/val_f1.png')
    # plt.clf()
