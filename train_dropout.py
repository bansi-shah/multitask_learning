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
parser.add_argument('--num_epochs', type=int, default=121, help='Number of epochs to train for')
parser.add_argument('--epoch_start_i', type=int, default=0, help='Start counting epochs from this number')
parser.add_argument('--checkpoint_step', type=int, default=15, help='How often to save checkpoints (epochs)')
parser.add_argument('--validation_step', type=int, default=1, help='How often to perform validation (epochs)')
parser.add_argument('--image', type=str, default=None, help='The image you want to predict on. Only valid in "predict" mode.')
parser.add_argument('--continue_training', type=str2bool, default=False, help='Whether to continue training from a checkpoint')
parser.add_argument('--dataset', type=str, default="CamVid", help='Dataset you are using.')
parser.add_argument('--crop_height', type=int, default=512, help='Height of cropped input image to network')
parser.add_argument('--crop_width', type=int, default=512, help='Width of cropped input image to network')
parser.add_argument('--batch_size'  , type=int, default=1, help='Number of images in each batch')
parser.add_argument('--num_val_images', type=int, default=40, help='The number of images to used for validations')
parser.add_argument('--h_flip', type=str2bool, default=False, help='Whether to randomly flip the image horizontally for data augmentation')
parser.add_argument('--v_flip', type=str2bool, default=False, help='Whether to randomly flip the image vertically for data augmentation')
parser.add_argument('--brightness', type=float, default=None, help='Whether to randomly change the image brightness for data augmentation. Specifies the max bightness change as a factor between 0.0 and 1.0. For example, 0.1 represents a max brightness change of 10%% (+-).')
parser.add_argument('--rotation', type=float, default=None, help='Whether to randomly rotate the image for data augmentation. Specifies the max rotation angle in degrees.')
parser.add_argument('--model', type=str, default="PSPNetDrop", help='The model you are using. See model_builder.py for supported models')
parser.add_argument('--frontend', type=str, default="ResNet101", help='The frontend you are using. See frontend_builder.py for supported models')
parser.add_argument('--checkpoint_path', type=str, default="auto_dropout_new", help='The path to save the checkpoint')
parser.add_argument('--log_file', type=str, default=None, help='The path to save logs')
args = parser.parse_args()

def data_augmentation(image, label1, label2, args):
    # Data augmentation
    if (args.crop_width <= image.shape[1]) and (args.crop_height <= image.shape[0]):
        x = random.randint(0, image.shape[1]-args.crop_width)
        y = random.randint(0, image.shape[0]-args.crop_height)
        
        if len(label2.shape) == 3:
            return image[y:y+args.crop_height, x:x+args.crop_width, :], label1[y:y+args.crop_height, x:x+args.crop_width, :],  label2[y:y+args.crop_height, x:x+args.crop_width, :], 
        else:
            return image[y:y+args.crop_height, x:x+args.crop_width, :], label1[y:y+args.crop_height, x:x+args.crop_width, :], label2[y:y+args.crop_height, x:x+args.crop_width], 
    else:
        raise Exception('Crop shape (%d, %d) exceeds image dimensions (%d, %d)!' % (args.crop_height, args.crop_width, image.shape[0], image.shape[1]))

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
prev_loss = 100000000000
threshold = 900
dropout = 0.40
default_value = 1000

import pickle as pkl
complexity_selective = pkl.load(open('complexity_selectivesearch.pkl', 'rb'))

# Compute your softmax cross entropy loss
net_input = tf.placeholder(tf.float32,shape=[None,None,None,3])
net_output1 = tf.placeholder(tf.float32,shape=[None,None,None,num_classes])
net_output2 = tf.placeholder(tf.float32,shape=[None,None,None,1])
dropout_rate = tf.Variable(0.6, name='dropout_rate')

network, init_fn = model_builder.build_model(model_name=args.model, frontend=args.frontend, net_input=net_input, num_classes=num_classes, crop_width=args.crop_width, crop_height=args.crop_height, is_training=True, dropout_rate=dropout_rate)
network1, network2 = network[0], network[1]

loss1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = network1, labels = net_output1))
loss2 = tf.reduce_mean(tf.losses.mean_squared_error(labels = net_output2, predictions = network2))

loss_param1 = tf.Variable(1.0, name='loss_param1')
loss_param2 = tf.Variable(1.0, name='loss_param2')
final_loss = loss_param1*loss1 + loss_param2*loss2

opt = tf.train.RMSPropOptimizer(learning_rate=0.0001, decay=0.995).minimize(final_loss, var_list=[var for var in tf.trainable_variables()])

saver=tf.train.Saver(max_to_keep=1000)
sess.run(tf.global_variables_initializer())

utils.count_params()

# If a pre-trained ResNet is required, load the weights.
# This must be done AFTER the variables are initialized with sess.run(tf.global_variables_initializer())
if init_fn is not None:
    init_fn(sess)

# Load a previous checkpoint if desired
model_checkpoint_name = args.checkpoint_path+"/latest_model_" + args.model + "_" + args.dataset + ".ckpt"
if args.continue_training:
    print('Loaded latest model checkpoint')
    saver.restore(sess, model_checkpoint_name)

# Load the data
print("Loading the data ...")
train_input_names, train_output1_names, train_output2_names, val_input_names, val_output1_names, val_output2_names, test_input_names, test_output1_names, test_output2_names = utils.prepare_data_multi(dataset=args.dataset)

print("\n***** Begin training *****")
print("Dataset -->", args.dataset)
print("Model -->", args.model)
print("Crop Height -->", args.crop_height)
print("Crop Width -->", args.crop_width)
print("Num Epochs -->", args.num_epochs)
print("Batch Size -->", args.batch_size)
print("Num Classes -->", num_classes)

print("Data Augmentation:")
print("\tVertical Flip -->", args.v_flip)
print("\tHorizontal Flip -->", args.h_flip)
print("\tBrightness Alteration -->", args.brightness)
print("\tRotation -->", args.rotation)
print("")

avg_loss_per_epoch = []
avg_loss1_per_epoch = []
avg_loss2_per_epoch = []
avg_scores_per_epoch = []
avg_iou_per_epoch = []
avg_validation_loss_per_epoch = []
avg_binary_loss_per_epoch = []
avg_time = []

# Which validation images do we want
val_indices = []
num_vals = min(args.num_val_images, len(val_input_names))

# Set random seed to make sure models are validated on the same validation images.
# So you can compare the results of different models more intuitively.
random.seed(16)
val_indices=random.sample(range(0,len(val_input_names)), num_vals)

'''
for i in range(len(train_input_names)):
  img = utils.load_image(train_input_names[i])
  print(train_input_names[i]) 
  x = utils.calculate_complexity(img)
 

for i in range(len(val_input_names)):
  img = utils.load_image(val_input_names[i])
  print(train_input_names[i]) 
  x = utils.calculate_complexity(img)


exit() 
'''

# Do the training here
for epoch in range(args.epoch_start_i, args.num_epochs):
    current_losses = []
    current_losses1 = []
    current_losses2 = []
    cnt=0

    # Equivalent to shuffling
    id_list = np.random.permutation(len(train_input_names))
    
    num_iters = int(np.floor(len(id_list) / args.batch_size))
    st = time.time()
    epoch_st=time.time()
    for i in range(num_iters):
        # st=time.time()
        input_image_batch = []
        output1_image_batch = []
        output2_image_batch = []

        # Collect a batch of images
        for j in range(args.batch_size):
            index = i*args.batch_size + j
            id = id_list[index]
            
            input_image = utils.load_image(train_input_names[id])
            output1_image = utils.load_image(train_output1_names[id])
            output2_image = utils.load_image_output(train_output2_names[id])

            with tf.device('/cpu:0'):
                input_image, output1_image, output2_image = data_augmentation(input_image, output1_image, output2_image, args)

                # Prep the data. Make sure the labels are in one-hot format
                input_image = np.float32(input_image) / 255.0
                output1_image = np.float32(helpers.one_hot_it(label=output1_image, label_values=label_values))
                output2_image = np.float32(np.expand_dims(output2_image, axis = -1)) / 255.0

                input_image_batch.append(np.expand_dims(input_image, axis=0))
                output1_image_batch.append(np.expand_dims(output1_image, axis=0))
                output2_image_batch.append(output2_image)
       
        input_image_batch = input_image_batch[0]
        output1_image_batch = output1_image_batch[0]

        new_rate = complexity_selective.get(train_input_names[id], default_value)
        if new_rate < threshold:
            new_rate = dropout
        else:
            new_rate = 1 - dropout
        a3 = dropout_rate.assign(new_rate)
        sess.run(a3)

        # Do the training
        _, current1, current2 = sess.run([opt, loss1, loss2], feed_dict = {net_input:input_image_batch, net_output1 : output1_image_batch, net_output2 : output2_image_batch})
        
        current_losses.append(current1+current2)
        current_losses1.append(current1)
        current_losses2.append(current2)
        
        cnt = cnt + args.batch_size
        if cnt % 20 == 0:
            string_print = "Epoch = %d Count = %d Current_Loss1 = %.4f Current_Loss2 = %.4f Time = %.2f "%(epoch, cnt, current1, current2, time.time()-st)
            utils.LOG(string_print, args.log_file)
            st = time.time()

    mean_loss = np.mean(current_losses)
    avg_loss_per_epoch.append(mean_loss)
    mean_loss1 = np.mean(current_losses1)
    avg_loss1_per_epoch.append(mean_loss1)
    mean_loss2 = np.mean(current_losses2)
    avg_loss2_per_epoch.append(mean_loss2)

    a1 = loss_param1.assign(mean_loss1/(mean_loss1+mean_loss2))
    a2 = loss_param2.assign(mean_loss2/(mean_loss1+mean_loss2))
    sess.run([a1, a2])

    # Create directories if needed
    if not os.path.isdir("%s/%04d"%(args.checkpoint_path,epoch)):
        os.makedirs("%s/%04d"%(args.checkpoint_path,epoch))

    # Save latest checkpoint to same file name
    print("Saving latest checkpoint")
    saver.save(sess, model_checkpoint_name)

    if val_indices != 0 and epoch % args.checkpoint_step == 0:
        print("Saving checkpoint for this epoch")
        saver.save(sess,"%s/%04d/model.ckpt"%(args.checkpoint_path, epoch))

    if epoch % args.validation_step == 0:
        print("Performing validation")
        target=open("%s/%04d/val_scores.csv"%(args.checkpoint_path, epoch),'w')
        target.write("val_name, avg_accuracy, precision, recall, f1 score, mean iou, %s\n" % (class_names_string))

        scores_list = []
        class_scores_list = []
        precision_list = []
        recall_list = []
        f1_list = []
        iou_list = []
        validation_loss_list = []
        binary_loss_list = []
        
        # Do the validation on a small set of validation images
        for ind in val_indices:
            img = utils.load_image(val_input_names[ind])[:args.crop_height, :args.crop_width]
            input_image = np.expand_dims(np.float32(img), axis=0)/255.0
            
            gt1 = utils.load_image(val_output1_names[ind])[:args.crop_height, :args.crop_width]
            gt1 = helpers.reverse_one_hot(helpers.one_hot_it(gt1, label_values))
            
            gt2 = utils.load_image_output(val_output2_names[ind])[:args.crop_height, :args.crop_width]
            gt2 = np.float32(np.expand_dims(gt2, axis = -1))            
            # st = time.time()
           
            new_rate = complexity_selective.get(train_input_names[id], default_value)
  
            #new_rate = utils.calculate_complexity(img)
            if new_rate < threshold:
               new_rate = dropout
            else:
               new_rate = 1 - dropout
            
            a3 = dropout_rate.assign(new_rate)
            sess.run(a3)

            output1_image, output2_image = sess.run([network1, network2], feed_dict={net_input:input_image})
            output1_image = np.array(output1_image[0,:,:,:])
            output1_image = helpers.reverse_one_hot(output1_image)
            out1_vis_image = helpers.colour_code_segmentation(output1_image, label_values)
            
            out2_vis_image = np.array(output2_image[0,:,:,:])*255

            accuracy, class_accuracies, prec, rec, f1, iou = utils.evaluate_segmentation(pred=output1_image, label=gt1, num_classes=num_classes)
            validation_loss, binary_loss = utils.evaluate_regression(pred=output2_image, label=gt2/255.0)

            file_name = utils.filepath_to_name(val_input_names[ind])
            target.write("%s, %f, %f, %f, %f, %f"%(file_name, accuracy, prec, rec, f1, iou))
            for item in class_accuracies:
                target.write(", %f"%(item))
            target.write("\n")

            scores_list.append(accuracy)
            class_scores_list.append(class_accuracies)
            precision_list.append(prec)
            recall_list.append(rec)
            f1_list.append(f1)
            iou_list.append(iou)
            validation_loss_list.append(validation_loss)
            binary_loss_list.append(binary_loss)
            gt1 = helpers.colour_code_segmentation(gt1, label_values)

            file_name = os.path.basename(val_input_names[ind])
            file_name = os.path.splitext(file_name)[0]
            # cv2.imwrite("%s/%04d/%s_org.png"%(args.checkpoint_path, epoch, file_name),cv2.cvtColor(np.uint8(input_image[0]*255.0), cv2.COLOR_RGB2BGR))
            # cv2.imwrite("%s/%04d/%s_pred1.png"%(args.checkpoint_path, epoch, file_name),cv2.cvtColor(np.uint8(out1_vis_image), cv2.COLOR_RGB2BGR))
            # cv2.imwrite("%s/%04d/%s_gt1.png"%(args.checkpoint_path, epoch, file_name),cv2.cvtColor(np.uint8(gt1), cv2.COLOR_RGB2BGR))
            # cv2.imwrite("%s/%04d/%s_pred2.png"%(args.checkpoint_path, epoch, file_name),cv2.cvtColor(np.uint8(out2_vis_image), cv2.COLOR_RGB2BGR))
            # cv2.imwrite("%s/%04d/%s_gt2.png"%(args.checkpoint_path, epoch, file_name),cv2.cvtColor(np.uint8(gt2), cv2.COLOR_GRAY2BGR))

        target.close()

        avg_score = np.mean(scores_list)
        class_avg_scores = np.mean(class_scores_list, axis=0)
        avg_scores_per_epoch.append(avg_score)
        avg_precision = np.mean(precision_list)
        avg_recall = np.mean(recall_list)
        avg_f1 = np.mean(f1_list)
        avg_iou = np.mean(iou_list)
        avg_iou_per_epoch.append(avg_iou)
        avg_validation_loss = np.mean(validation_loss_list)
        avg_validation_loss_per_epoch.append(avg_validation_loss)
        avg_binary_loss = np.mean(binary_loss_list)
        avg_binary_loss_per_epoch.append(avg_binary_loss)

        print("\nAverage validation accuracy for epoch # %04d = %f"% (epoch, avg_score))
        print("Average per class validation accuracies for epoch # %04d:"% (epoch))
        for index, item in enumerate(class_avg_scores):
            print("%s = %f" % (class_names_list[index], item))
        print("Validation precision = ", avg_precision)
        print("Validation recall = ", avg_recall)
        print("Validation F1 score = ", avg_f1)
        print("Validation IoU score = ", avg_iou)
        print("Validation Regression Loss score = ", avg_validation_loss)
        print("Validation Binary Loss score =", avg_binary_loss)

    epoch_time=time.time()-epoch_st
    remain_time=epoch_time*(args.num_epochs-1-epoch)
    m, s = divmod(remain_time, 60)
    h, m = divmod(m, 60)
    if s!=0:
        train_time="Remaining training time : %d hours %d minutes %d seconds\n"%(h,m,s)
    else:
        train_time="Remaining training time : Training completed.\n"
    utils.LOG(train_time, args.log_file)
    avg_time.append(epoch_time)
    scores_list = []

    fig1, ax1 = plt.subplots(figsize=(11, 8))   
    ax1.plot(range(args.epoch_start_i, epoch+1), avg_scores_per_epoch)
    ax1.set_title("Validation average accuracy vs epochs")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Avg. val. accuracy")
    plt.savefig(args.checkpoint_path+ '/val_accuracy_vs_epochs.png')
    plt.clf()

    fig2, ax2 = plt.subplots(figsize=(11, 8))
    ax2.plot(range(args.epoch_start_i, epoch+1), avg_loss_per_epoch)
    ax2.set_title("Train Overall loss vs epochs")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Current loss")
    plt.savefig(args.checkpoint_path +'/train_loss_vs_epochs.png')
    plt.clf()

    fig2, ax2 = plt.subplots(figsize=(11, 8))
    ax2.plot(range(args.epoch_start_i, epoch+1), avg_loss2_per_epoch)
    ax2.set_title("Train Regression loss vs epochs")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Current loss")
    plt.savefig(args.checkpoint_path +'/train_regression_loss_vs_epochs.png')
    plt.clf()

    fig2, ax2 = plt.subplots(figsize=(11, 8))
    ax2.plot(range(args.epoch_start_i, epoch+1), avg_loss1_per_epoch)
    ax2.set_title("Train Segmentation loss vs epochs")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Current loss")
    plt.savefig(args.checkpoint_path +'/train_segmentation_loss_vs_epochs.png')
    plt.clf()

    fig3, ax3 = plt.subplots(figsize=(11, 8))
    ax3.plot(list(range(args.epoch_start_i, epoch+1)), avg_iou_per_epoch)
    ax3.set_title("Validation average IoU vs epochs")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Current IoU")
    plt.savefig(args.checkpoint_path + '/val_iou_vs_epochs.png')
    plt.clf()

    fig1, ax1 = plt.subplots(figsize=(11, 8))
    ax1.plot(range(args.epoch_start_i, epoch+1), avg_validation_loss_per_epoch)
    ax1.set_title("Validation average regression loss vs epochs")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Avg. val. loss")
    plt.savefig(args.checkpoint_path+ '/val_regression_loss_vs_epochs.png')
    plt.clf()

    fig1, ax1 = plt.subplots(figsize=(11, 8))
    ax1.plot(range(args.epoch_start_i, epoch+1), avg_binary_loss_per_epoch)
    ax1.set_title("Validation average binary loss vs epochs")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Avg. val. loss")
    plt.savefig(args.checkpoint_path+ '/val_binary_loss_vs_epochs.png')
    plt.clf()

    # fig1, ax1 = plt.subplots(figsize=(11, 8))
    # ax1.plot(range(args.epoch_start_i, epoch+1), avg_time)
    # ax1.set_title("Time vs Epochs")
    # ax1.set_xlabel("Epoch")
    # ax1.set_ylabel("time")
    # plt.savefig(args.checkpoint_path+ '/time_vs_epochs.png')
    # plt.clf()
