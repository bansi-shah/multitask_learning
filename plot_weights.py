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
parser.add_argument('--num_epochs', type=int, default=0, help='Number of epochs to train for')
parser.add_argument('--epoch_start_i', type=int, default=0, help='Start counting epochs from this number')
parser.add_argument('--checkpoint_step', type=int, default=15, help='How often to save checkpoints (epochs)')
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
parser.add_argument('--model', type=str, default="PSPNetRes", help='The model you are using. See model_builder.py for supported models')
parser.add_argument('--frontend', type=str, default="ResNet101", help='The frontend you are using. See frontend_builder.py for supported models')
parser.add_argument('--checkpoint_path', type=str, default="psp_camvid_multi_new_difficult_res4", help='The path to save the checkpoint')
parser.add_argument('--log_file', type=str, default=None, help='The path to save logs')
parser.add_argument('--resize', type=bool, default=False, help='Crop or resize image')

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
network1, network2 = network[0], network[1]

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
#train_input_names, train_output1_names, train_output2_names, val_input_names, val_output1_names, val_output2_names, test_input_names, test_output1_names, test_output2_names = utils.get_simple_task(dataset=args.dataset)

for v in tf.trainable_variables():
  var_vals = sess.run(v)
  name = '0015/' + v.name.replace('/', '-')+'('+','.join(map(str, var_vals.shape))+')'
  print(name)
  np.save(name+'.npy', var_vals)


scores_list = []
class_scores_list = []
precision_list = []
recall_list = []
f1_list = []
iou_list = []
validation_loss_list = []
binary_loss_list = []

for ind in range(len(test_input_names)):
    input_image = np.expand_dims(np.float32(utils.load_image(test_input_names[ind])[:args.crop_height, :args.crop_width]),axis=0)/255.0

    gt1 = utils.load_image(test_output1_names[ind])[:args.crop_height, :args.crop_width]
    gt1 = helpers.reverse_one_hot(helpers.one_hot_it(gt1, label_values))

    gt2 = utils.load_image_output(test_output2_names[ind])[:args.crop_height, :args.crop_width]
    gt2 = np.float32(np.expand_dims(gt2, axis = -1))            
        # st = time.time()

    output1_image, output2_image = sess.run([network1, network2], feed_dict={net_input:input_image})
    output1_image = np.array(output1_image[0,:,:,:])
    output1_image = helpers.reverse_one_hot(output1_image)

    output2_image = np.array(output2_image[0,:,:,:])
    
    accuracy, class_accuracies, prec, rec, f1, iou = utils.evaluate_segmentation(pred=output1_image, label=gt1, num_classes=num_classes)
    validation_loss, binary_loss = utils.evaluate_regression(pred=output2_image, label=gt2/255.0)

    scores_list.append(accuracy)
    class_scores_list.append(class_accuracies)
    precision_list.append(prec)
    recall_list.append(rec)
    f1_list.append(f1)
    iou_list.append(iou)
    validation_loss_list.append(validation_loss)
    binary_loss_list.append(binary_loss)
    
    out1_vis_image = helpers.colour_code_segmentation(output1_image, label_values)
    out2_vis_image = np.array(output2_image*255.0)

    file_name = os.path.basename(train_input_names[ind])
    file_name = os.path.splitext(file_name)[0]
    
    # cv2.imwrite("%s/%s_org.png"%("Test2", file_name),cv2.cvtColor(np.uint8(input_image[0]*255.0), cv2.COLOR_RGB2BGR))
    # cv2.imwrite("%s/%s_pred1.png"%("Test2", file_name),cv2.cvtColor(np.uint8(out1_vis_image), cv2.COLOR_RGB2BGR))
    # cv2.imwrite("%s/%s_gt1.png"%("Test2", file_name),cv2.cvtColor(np.uint8(gt1), cv2.COLOR_RGB2BGR))
    # cv2.imwrite("%s/%s_pred2.png"%("Test2", file_name),cv2.cvtColor(np.uint8(out2_vis_image), cv2.COLOR_GRAY2BGR))
    # cv2.imwrite("%s/%s_gt2.png"%("Test2", file_name),cv2.cvtColor(np.uint8(gt2), cv2.COLOR_GRAY2BGR))

avg_score = np.mean(scores_list)
class_avg_scores = np.mean(class_scores_list, axis=0)
avg_precision = np.mean(precision_list)
avg_recall = np.mean(recall_list)
avg_f1 = np.mean(f1_list)
avg_iou = np.mean(iou_list)
avg_validation_loss = np.mean(validation_loss_list)
avg_binary_loss = np.mean(binary_loss_list)

print("\nAverage validation accuracy %.4f"% (avg_score))
print("Average per class validation accuracies ")
for index, item in enumerate(class_avg_scores):
    print("%s = %.4f" % (class_names_list[index], item))
print("Test precision = ", avg_precision)
print("Test recall = ", avg_recall)
print("Test F1 score = ", avg_f1)
print("Test IoU score = ", avg_iou)
print("Test Regression Loss score = ", avg_validation_loss)
print("Test Binary Loss score =", avg_binary_loss)
