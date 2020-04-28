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
parser.add_argument('--batch_size', type=int, default=8, help='Number of images in each batch')
parser.add_argument('--model', type=str, default="PSPNetMultiExp", help='The model you are using. See model_builder.py for supported models')
parser.add_argument('--frontend', type=str, default="ResNet101", help='The frontend you are using. See frontend_builder.py for supported models')
parser.add_argument('--checkpoint_path', type=str, default="camvid_multitask_new_3", help='The path to save the checkpoint')
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

# Compute your softmax cross entropy loss
net_input = tf.placeholder(tf.float32,shape=[None, None, None, 3])
net_output1 = tf.placeholder(tf.float32,shape=[None, None, None, num_classes])
net_output2 = tf.placeholder(tf.float32,shape=[None, None, None,1])
net_output3 = tf.placeholder(tf.float32,shape=[None, None, None, 2]) #fg, bg

network, init_fn = model_builder.build_model(model_name=args.model, frontend=args.frontend, net_input=net_input, num_classes=num_classes, crop_width=args.crop_width, crop_height=args.crop_height, is_training=True)
network1, network2, network3 = network[0], network[1], network[2]

saver=tf.train.Saver(max_to_keep=1000)
sess.run(tf.global_variables_initializer())

utils.count_params()

if init_fn is not None:
    init_fn(sess)

# Load a previous checkpoint if desired
model_checkpoint_name = args.checkpoint_path+"/latest_model_" + args.model + "_" + args.dataset + ".ckpt"
print('Loaded latest model checkpoint')
saver.restore(sess, model_checkpoint_name)

# Load the data
print("Loading the data ...")
train_input_names, train_output1_names, train_output2_names, val_input_names, val_output1_names, val_output2_names, test_input_names, test_output1_names, test_output2_names = utils.prepare_data_multi(dataset=args.dataset)

print("Dataset -->", args.dataset)
print("Model -->", args.model)

# os.makedirs('Test', exist_ok=True)

scores_list = []
class_scores_list = []
precision_list = []
recall_list = []
f1_list = []
iou_list = []

validation_loss_list = []
binary_loss_list = []

scores_list_2 = []
class_scores_list_2 = []
precision_list_2 = []
recall_list_2 = []
f1_list_2 = []
iou_list_2 = []

for ind in range(len(test_input_names)):
    input_image = np.expand_dims(np.float32(utils.load_image(test_input_names[ind])[:args.crop_height, :args.crop_width]),axis=0)/255.0
    gt1 = utils.load_image(test_output1_names[ind])[:args.crop_height, :args.crop_width]
    gt1 = helpers.reverse_one_hot(helpers.one_hot_it(gt1, label_values))

    gt2 = utils.load_image_output(test_output2_names[ind])[:args.crop_height, :args.crop_width]
    gt2 = np.float32(np.expand_dims(gt2, axis = -1))            
    # st = time.time()

    gt3 = utils.load_image(test_output1_names[ind])[:args.crop_height, :args.crop_width]
    gt3 =  helpers.reverse_one_hot(helpers.fgbg(label = gt3, label_values = label_values, class_names = class_names_list))
    
    output1_image = sess.run(network1, feed_dict={net_input:input_image})
    output1_image = np.array(output1_image[0,:,:,:])
    output1_image = helpers.reverse_one_hot(output1_image)
    out1_vis_image = helpers.colour_code_segmentation(output1_image, label_values)
    
    output2_image = sess.run(network2, feed_dict={net_input:input_image})
    out2_vis_image = np.array(output2_image[0,:,:,:])*255

    output3_image = sess.run(network3, feed_dict={net_input:input_image})
    output3_image = helpers.reverse_one_hot(output3_image[0, :, :, :])         
    out3_vis_image = helpers.colour_code_fgbg(output3_image)
    
    accuracy, class_accuracies, prec, rec, f1, iou = utils.evaluate_segmentation(pred=output1_image, label=gt1, num_classes=num_classes)
    validation_loss, binary_loss = utils.evaluate_regression(pred=output2_image, label=gt2/255.0)
    accuracy_2, class_accuracies_2, prec_2, rec_2, f1_2, iou_2 = utils.evaluate_segmentation(pred=output3_image, label = gt3, num_classes=2)
    
    scores_list.append(accuracy)
    class_scores_list.append(class_accuracies)
    precision_list.append(prec)
    recall_list.append(rec)
    f1_list.append(f1)
    iou_list.append(iou)

    validation_loss_list.append(validation_loss)
    binary_loss_list.append(binary_loss)

    scores_list_2.append(accuracy_2)
    class_scores_list_2.append(class_accuracies_2)
    precision_list_2.append(prec_2)
    recall_list_2.append(rec_2)
    f1_list_2.append(f1_2)
    iou_list_2.append(iou_2)

    file_name = os.path.basename(test_input_names[ind])
    file_name = os.path.splitext(file_name)[0]

    # cv2.imwrite("%s/%s_org.png"%("Test", file_name),cv2.cvtColor(np.uint8(input_image[0]*255.0), cv2.COLOR_RGB2BGR))
    # cv2.imwrite("%s/%s_pred1.png"%("Test", file_name),cv2.cvtColor(np.uint8(out1_vis_image), cv2.COLOR_RGB2BGR))
    # cv2.imwrite("%s/%s_gt1.png"%("Test", file_name),cv2.cvtColor(np.uint8(gt1), cv2.COLOR_RGB2BGR))
    # cv2.imwrite("%s/%s_pred2.png"%("Test", file_name),cv2.cvtColor(np.uint8(out2_vis_image), cv2.COLOR_RGB2BGR))
    # cv2.imwrite("%s/%s_gt2.png"%("Test", file_name),cv2.cvtColor(np.uint8(gt2), cv2.COLOR_GRAY2BGR))
    # cv2.imwrite("%s/%s_pred3.png"%("Test", file_name),cv2.cvtColor(np.uint8(out3_vis_image), cv2.COLOR_RGB2BGR))
    # cv2.imwrite("%s/%s_gt3.png"%("Test", file_name),cv2.cvtColor(np.uint8(gt3), cv2.COLOR_GRAY2BGR))


avg_score = np.mean(scores_list)
class_avg_scores = np.mean(class_scores_list, axis=0)
avg_precision = np.mean(precision_list)
avg_recall = np.mean(recall_list)
avg_f1 = np.mean(f1_list)
avg_iou = np.mean(iou_list)

avg_validation_loss = np.mean(validation_loss_list)
avg_binary_loss = np.mean(binary_loss_list)

avg_score_2 = np.mean(scores_list_2)
class_avg_scores_2 = np.mean(class_scores_list_2, axis=0)
avg_precision_2 = np.mean(precision_list_2)
avg_recall_2 = np.mean(recall_list_2)
avg_f1_2 = np.mean(f1_list_2)
avg_iou_2 = np.mean(iou_list_2)

print("\nAverage validation accuracy = %f"% (avg_score))
print("Average per class validation accuracies")
for index, item in enumerate(class_avg_scores):
    print("\t%s = %f" % (class_names_list[index], item))
print("Validation precision = ", avg_precision)
print("Validation recall = ", avg_recall)
print("Validation F1 score = ", avg_f1)
print("Validation IoU score = ", avg_iou)

print("Validation Regression Loss score = ", avg_validation_loss)
print("Validation Binary Loss score =", avg_binary_loss)

print("\nAverage validation accuracy = %f"% (avg_score_2))
print("Validation precision2 = ", avg_precision_2)
print("Validation recall2 = ", avg_recall_2)
print("Validation F1 score2 = ", avg_f1_2)
print("Validation IoU score2 = ", avg_iou_2)
