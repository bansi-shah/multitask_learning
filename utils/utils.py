from __future__ import print_function, division
import os,time,cv2, sys, math
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import time, datetime
import os, random
from cv2 import imread
import ast
from sklearn.metrics import precision_score, \
    recall_score, confusion_matrix, classification_report, \
    accuracy_score, f1_score

from utils import helpers, conf
from utils.conf import *

import scipy
from scipy import ndimage
import selectivesearch
from selectivesearch import selective_search

def calculate_complexity(image):
    l, regions = selective_search(image, scale=500, sigma=0.9, min_size=20)
    complexity = len(regions)
    print(complexity)
    '''
    dy = ndimage.filters.sobel(image, 1)
    dx = ndimage.filters.sobel(image, 0) 
    map_si =  dx*dx + dy*dy
    # si_mean = np.mean( map_si )
    # si_rms  = np.sqrt( 1.0/npix * np.sum( map_si * map_si )  )
    # si_std  = np.std( map_si )
    map_si = map_si != 0
    npix   = map_si.shape[0] * map_si.shape[1] * map_si.shape[2]
    complexity = np.count_nonzero( map_si ) / np.float32( npix ) 
    '''
    return complexity

def get_train_val_data_2(dataset_dir,  isvalid):
    train_input_names=[]
    train_output_names=[]
    val_input_names=[]
    val_output_names=[]
    
    for cities in os.listdir(dataset_dir + "/train"):
        for file in os.listdir(dataset_dir + "/train/" + cities):
            cwd = os.getcwd()
            train_output_names.append(cwd + "/" + dataset_dir + "/train_labels/" + cities + "/" + file[:-15] +"gtFine_color.png" )
            train_input_names.append(cwd + "/" + dataset_dir + "/train/" + cities + "/" + file)
    
    for cities in os.listdir(dataset_dir + "/val"):
        if isvalid:
            for file in os.listdir(dataset_dir + "/val/" + cities):
                cwd = os.getcwd()
                val_input_names.append(cwd + "/" + dataset_dir + "/val/" + cities + "/" + file)
                val_output_names.append(cwd + "/" + dataset_dir + "/val_labels/" + cities + "/" + file[:-15] +"gtFine_color.png" )
                # print(val_input_names[-1], val_output_names[-1])
        else:
            x = (int)(split_ratio * len(train_input_names))
            return train_input_names[:x], train_output_names[:x], train_input_names[x:], train_output_names[x:] 
    return train_input_names,train_output_names, val_input_names, val_output_names

def get_test_data_2(dataset_dir):
    test_input_names=[]
    test_output_names=[]
    for file in os.listdir(dataset_dir + "/test"):
        for cities in os.listdir(dataset_dir + "/test"):
            cwd = os.getcwd()
            test_input_names.append(cwd + "/" + dataset_dir + "/test/" + cities + "/" + file)
            test_output_names.append(cwd + "/" + dataset_dir + "/test_labels/" + cities + "/" + file[:-15]+"gtFine_color.png")

    return test_input_names, test_output_names


def get_train_val_data(dataset_dir, isvalid = False):
    if 'cityscapes' in dataset_dir:
        return get_train_val_data_2(dataset_dir, isvalid)
    
    train_input_names=[]
    train_output_names=[]
    val_input_names=[]
    val_output_names=[]

    # for file in os.listdir(dataset_dir + "/train"):
    #     cwd = os.getcwd()
    #     train_input_names.append(cwd + "/" + dataset_dir + "/train/" + file)
    for file in os.listdir(dataset_dir + "/train"):
        cwd = os.getcwd()
        train_output_names.append(cwd + "/" + dataset_dir + "/train_labels/" + file)
        train_input_names.append(cwd+"/" + dataset_dir + "/train/" + file)

    if isvalid:
        for file in os.listdir(dataset_dir + "/val"):
            cwd = os.getcwd()
            val_input_names.append(cwd + "/" + dataset_dir + "/val/" + file)
        for file in os.listdir(dataset_dir + "/val_labels"):
            cwd = os.getcwd()
            val_output_names.append(cwd + "/" + dataset_dir + "/val_labels/" + file)
    else:
        x = (int)(split_ratio * len(train_input_names))
        return train_input_names[:x], train_output_names[:x], train_input_names[x:], train_output_names[x:] 
    return train_input_names,train_output_names, val_input_names, val_output_names

def get_test_data(dataset_dir):
    if 'cityscapes' in dataset_dir:
        return get_test_data_2(dataset_dir)
   
    test_input_names=[]
    test_output_names=[]
    for file in os.listdir(dataset_dir + "/test"):
        cwd = os.getcwd()
        test_input_names.append(cwd + "/" + dataset_dir + "/test/" + file)
    for file in os.listdir(dataset_dir + "/test_labels"):
        cwd = os.getcwd()
        test_output_names.append(cwd + "/" + dataset_dir + "/test_labels/" + file)

    return test_input_names, test_output_names

def prepare_data(dataset):
    train_input_names,train_output_names, val_input_names, val_output_names = get_train_val_data(dataset_dir[dataset]['path'], dataset_dir[dataset]['isvalid'])
    test_input_names, test_output_names = get_test_data(dataset_dir[dataset]['path'])
    train_input_names.sort(),train_output_names.sort(), val_input_names.sort(), val_output_names.sort(), test_input_names.sort(), test_output_names.sort()
    return train_input_names,train_output_names, val_input_names, val_output_names, test_input_names, test_output_names

def prepare_data_multi(dataset):
    train_input_names, train_output1_names, val_input_names, val_output1_names, test_input_names, test_output1_names = prepare_data(dataset)

    train_output2_names, val_output2_names, test_output2_names=[], [], []

    dataset_dir_ = dataset_dir[dataset]['path']

    for file in os.listdir(dataset_dir_ + "/test_sal_newdata_s75"):
        cwd = os.getcwd()
        test_output2_names.append(cwd + "/" + dataset_dir_ + "/test_sal_newdata_s75/" + file)
    
    for file in os.listdir(dataset_dir_ + "/train_sal_newdata_s75"):
        cwd = os.getcwd()
        train_output2_names.append(cwd + "/" + dataset_dir_ + "/train_sal_newdata_s75/" + file)

    for file in os.listdir(dataset_dir_ + "/val_sal_newdata_s75"):
        cwd = os.getcwd()
        val_output2_names.append(cwd + "/" + dataset_dir_ + "/val_sal_newdata_s75/" + file)

    train_output2_names.sort()
    val_output2_names.sort()
    test_output2_names.sort()

    return train_input_names, train_output1_names, train_output2_names, val_input_names, val_output1_names, val_output2_names, test_input_names, test_output1_names, test_output2_names


def get_simple_task(dataset):
    dataset_dir_ = '../data/camvid_all/simple'
    seg_dir = dataset_dir[dataset]['path']  +  '/segmentation_labels/'
    input_dir = dataset_dir[dataset]['path'] + '/input/'

    train_input_names = []
    train_output1_names = []
    train_output2_names = []

    val_input_names = []
    val_output1_names = []
    val_output2_names = []
    
    test_input_names = []
    test_output1_names = []
    test_output2_names = []


    for file in os.listdir(dataset_dir_ + "/test_sal"):
        cwd = os.getcwd()
        test_output2_names.append(cwd + "/" + dataset_dir_ + "/test_sal/" + file)
        test_output1_names.append(cwd + "/" + seg_dir + file[:-4]+'_L.png')
        test_input_names.append(cwd + "/" + input_dir + file)
    
    for file in os.listdir(dataset_dir_ + "/train_sal"):
        cwd = os.getcwd()
        train_output2_names.append(cwd + "/" + dataset_dir_ + "/train_sal/" + file)
        train_output1_names.append(cwd + "/" + seg_dir + file[:-4]+'_L.png')
        train_input_names.append(cwd + "/" + input_dir + file)
    
    for file in os.listdir(dataset_dir_ + "/val_sal"):
        cwd = os.getcwd()
        val_output2_names.append(cwd + "/" + dataset_dir_ + "/val_sal/" + file)
        val_output1_names.append(cwd + "/" + seg_dir + file[:-4]+'_L.png')
        val_input_names.append(cwd + "/" + input_dir + file)
    
    train_input_names.sort()
    val_input_names.sort()
    test_input_names.sort()

    train_output1_names.sort()
    val_output1_names.sort()
    test_output1_names.sort()

    train_output2_names.sort()
    val_output2_names.sort()
    test_output2_names.sort()

    return train_input_names, train_output1_names, train_output2_names, val_input_names, val_output1_names, val_output2_names, test_input_names, test_output1_names, test_output2_names

def get_difficult_task(dataset):
    dataset_dir_ = '../data/camvid_all/difficult_new'
    seg_dir = dataset_dir[dataset]['path']  +  '/segmentation_labels/'
    input_dir = dataset_dir[dataset]['path'] + '/input/'

    train_input_names = []
    train_output1_names = []
    train_output2_names = []

    val_input_names = []
    val_output1_names = []
    val_output2_names = []
    
    test_input_names = []
    test_output1_names = []
    test_output2_names = []


    for file in os.listdir(dataset_dir_ + "/test_sal"):
        cwd = os.getcwd()
        test_output2_names.append(cwd + "/" + dataset_dir_ + "/test_sal/" + file)
        test_output1_names.append(cwd + "/" + seg_dir + file[:-4]+'_L.png')
        test_input_names.append(cwd + "/" + input_dir + file)
    
    for file in os.listdir(dataset_dir_ + "/train_sal"):
        cwd = os.getcwd()
        train_output2_names.append(cwd + "/" + dataset_dir_ + "/train_sal/" + file)
        train_output1_names.append(cwd + "/" + seg_dir + file[:-4]+'_L.png')
        train_input_names.append(cwd + "/" + input_dir + file)
    
    for file in os.listdir(dataset_dir_ + "/val_sal"):
        cwd = os.getcwd()
        val_output2_names.append(cwd + "/" + dataset_dir_ + "/val_sal/" + file)
        val_output1_names.append(cwd + "/" + seg_dir + file[:-4]+'_L.png')
        val_input_names.append(cwd + "/" + input_dir + file)
    
    train_input_names.sort()
    val_input_names.sort()
    test_input_names.sort()

    train_output1_names.sort()
    val_output1_names.sort()
    test_output1_names.sort()

    train_output2_names.sort()
    val_output2_names.sort()
    test_output2_names.sort()

    return train_input_names, train_output1_names, train_output2_names, val_input_names, val_output1_names, val_output2_names, test_input_names, test_output1_names, test_output2_names


def load_image(path):
    # print(path)
    image = cv2.cvtColor(cv2.imread(path,-1), cv2.COLOR_BGR2RGB)
    return image

def load_image_output(path):
    image = cv2.imread(path, 0)
    return image    

# Takes an absolute file path and returns the name of the file without th extension
def filepath_to_name(full_name):
    file_name = os.path.basename(full_name)
    file_name = os.path.splitext(file_name)[0]
    return file_name

# Print with time. To console or file
def LOG(X, f=None):
    time_stamp = datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    if not f:
        print(time_stamp + " " + X)
    else:
        with open(f, 'a') as f:
            f.write(time_stamp + " " + X)

# Count total number of parameters in the model
def count_params():
    total_parameters = 0
    for variable in tf.trainable_variables():
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    print("This model has %d trainable parameters"% (total_parameters))

# Subtracts the mean images from ImageNet
def mean_image_subtraction(inputs, means=[123.68, 116.78, 103.94]):
    inputs=tf.to_float(inputs)
    num_channels = inputs.get_shape().as_list()[-1]
    if len(means) != num_channels:
      raise ValueError('len(means) must match the number of channels')
    channels = tf.split(axis=3, num_or_size_splits=num_channels, value=inputs)
    for i in range(num_channels):
        channels[i] -= means[i]
    return tf.concat(axis=3, values=channels)

def _lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    gts = tf.reduce_sum(gt_sorted)
    intersection = gts - tf.cumsum(gt_sorted)
    union = gts + tf.cumsum(1. - gt_sorted)
    jaccard = 1. - intersection / union
    jaccard = tf.concat((jaccard[0:1], jaccard[1:] - jaccard[:-1]), 0)
    return jaccard

def _flatten_probas(probas, labels, ignore=None, order='BHWC'):
    """
    Flattens predictions in the batch
    """
    if order == 'BCHW':
        probas = tf.transpose(probas, (0, 2, 3, 1), name="BCHW_to_BHWC")
        order = 'BHWC'
    if order != 'BHWC':
        raise NotImplementedError('Order {} unknown'.format(order))
    C = probas.shape[3]
    probas = tf.reshape(probas, (-1, C))
    labels = tf.reshape(labels, (-1,))
    if ignore is None:
        return probas, labels
    valid = tf.not_equal(labels, ignore)
    vprobas = tf.boolean_mask(probas, valid, name='valid_probas')
    vlabels = tf.boolean_mask(labels, valid, name='valid_labels')
    return vprobas, vlabels

def _lovasz_softmax_flat(probas, labels, only_present=True):
    """
    Multi-class Lovasz-Softmax loss
      probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
      labels: [P] Tensor, ground truth labels (between 0 and C - 1)
      only_present: average only on classes present in ground truth
    """
    C = probas.shape[1]
    losses = []
    present = []
    for c in range(C):
        fg = tf.cast(tf.equal(labels, c), probas.dtype) # foreground for class c
        if only_present:
            present.append(tf.reduce_sum(fg) > 0)
        errors = tf.abs(fg - probas[:, c])
        errors_sorted, perm = tf.nn.top_k(errors, k=tf.shape(errors)[0], name="descending_sort_{}".format(c))
        fg_sorted = tf.gather(fg, perm)
        grad = _lovasz_grad(fg_sorted)
        losses.append(
            tf.tensordot(errors_sorted, tf.stop_gradient(grad), 1, name="loss_class_{}".format(c))
                      )
    losses_tensor = tf.stack(losses)
    if only_present:
        present = tf.stack(present)
        losses_tensor = tf.boolean_mask(losses_tensor, present)
    return losses_tensor

def lovasz_softmax(probas, labels, only_present=True, per_image=False, ignore=None, order='BHWC'):
    """
    Multi-class Lovasz-Softmax loss
      probas: [B, H, W, C] or [B, C, H, W] Variable, class probabilities at each prediction (between 0 and 1)
      labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
      only_present: average only on classes present in ground truth
      per_image: compute the loss per image instead of per batch
      ignore: void class labels
      order: use BHWC or BCHW
    """
    probas = tf.nn.softmax(probas, 3)
    labels = helpers.reverse_one_hot(labels)

    if per_image:
        def treat_image(prob, lab):
            prob, lab = tf.expand_dims(prob, 0), tf.expand_dims(lab, 0)
            prob, lab = _flatten_probas(prob, lab, ignore, order)
            return _lovasz_softmax_flat(prob, lab, only_present=only_present)
        losses = tf.map_fn(treat_image, (probas, labels), dtype=tf.float32)
    else:
        losses = _lovasz_softmax_flat(*_flatten_probas(probas, labels, ignore, order), only_present=only_present)
    return losses

def crop_data_video(input1_image, input2_image, input3_image, label1, label2, args):    
    crop_height, crop_width = args.crop_height, args.crop_width
    x = random.randint(0, input1_image.shape[1]-crop_width)
    y = random.randint(0, input1_image.shape[0]-crop_height)

    if len(label2.shape) == 3:
        return input1_image[y:y+crop_height, x:x+crop_width, :], input2_image[y:y+crop_height, x:x+crop_width, :], input3_image[y:y+crop_height, x:x+crop_width, :], label1[y:y+crop_height, x:x+crop_width, :], label2[y:y+crop_height, x:x+crop_width, :]
    else:
        return input1_image[y:y+crop_height, x:x+crop_width, :], input2_image[y:y+crop_height, x:x+crop_width, :], input3_image[y:y+crop_height, x:x+crop_width, :], label1[y:y+crop_height, x:x+crop_width, :], label2[y:y+crop_height, x:x+crop_width]

def data_augmentation(input_image, output_image, args):
    # Data augmentation
    if args.resize:
        input_image, output_image = resize(input_image, output_image, args.crop_height, args.crop_width)
    else:
        input_image, output_image = random_crop(input_image, output_image, args.crop_height, args.crop_width)

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

def resize(image, label, crop_height, crop_width):
    return cv2.resize(image, (crop_height, crop_width)), cv2.resize(label, (crop_height, crop_width))
    
# Randomly crop the image to a specific size. For data augmentation
def random_crop(image, label, crop_height, crop_width):
    # print(image.shape, label.shape)
    if (image.shape[0] != label.shape[0]) or (image.shape[1] != label.shape[1]):
        raise Exception('Image and label must have the same dimensions!')
        
    if (crop_width <= image.shape[1]) and (crop_height <= image.shape[0]):
        x = random.randint(0, image.shape[1]-crop_width)
        y = random.randint(0, image.shape[0]-crop_height)
        
        if len(label.shape) == 3:
            return image[y:y+crop_height, x:x+crop_width, :], label[y:y+crop_height, x:x+crop_width, :]
        else:
            return image[y:y+crop_height, x:x+crop_width, :], label[y:y+crop_height, x:x+crop_width]
    else:
        raise Exception('Crop shape (%d, %d) exceeds image dimensions (%d, %d)!' % (crop_height, crop_width, image.shape[0], image.shape[1]))

# Compute the average segmentation accuracy across all classes
def compute_global_accuracy(pred, label):
    total = len(label)
    count = 0.0
    for i in range(total):
        if pred[i] == label[i]:
            count = count + 1.0
    return float(count) / float(total)

# Compute the class-specific segmentation accuracy
def compute_class_accuracies(pred, label, num_classes):
    total = []
    for val in range(num_classes):
        total.append((label == val).sum())

    count = [0.0] * num_classes
    for i in range(len(label)):
        if pred[i] == label[i]:
            count[int(pred[i])] = count[int(pred[i])] + 1.0

    # If there are no pixels from a certain class in the GT, 
    # it returns NAN because of divide by zero
    # Replace the nans with a 1.0.
    accuracies = []
    for i in range(len(total)):
        if total[i] == 0:
            accuracies.append(1.0)
        else:
            accuracies.append(count[i] / total[i])

    return accuracies


def compute_mean_iou(pred, label):
    unique_labels = np.unique(label)
    num_unique_labels = len(unique_labels)
    
    I = np.zeros(num_unique_labels)
    U = np.zeros(num_unique_labels)

    for index, val in enumerate(unique_labels):
        pred_i = pred == val
        label_i = label == val

        I[index] = float(np.sum(np.logical_and(label_i, pred_i)))
        U[index] = float(np.sum(np.logical_or(label_i, pred_i)))

    mean_iou = np.mean(I / U)
    return mean_iou

def mse(pred, label):
    x = pred-label
    n = x.shape[0]
    return np.dot(x, x)/n

def mae(pred, label):
    x = pred-label
    return np.absolute(x)   

def evaluate_regression(pred, label, threshold = 0.01):
    pred = np.squeeze(pred).flatten()
    label = np.squeeze(label).flatten()
    x = pred-label
    y = (abs(x) <= threshold).sum()
    n = x.shape[0]
    return np.sqrt(np.dot(x, x))/n , y/pred.shape[0]
    
    # pred_ = (pred >= 0.5)
    # return mse(pred, label), mae(pred, label), precision_score(pred_, label, average="weighted"), recall_score(pred_, label, average="weighted"), f1_score(pred_, label, average="weighted")

def evaluate_segmentation(pred, label, num_classes, score_averaging="weighted"):
    flat_pred = pred.flatten()
    flat_label = label.flatten()

    global_accuracy = compute_global_accuracy(flat_pred, flat_label)
    class_accuracies = compute_class_accuracies(flat_pred, flat_label, num_classes)

    prec = precision_score(flat_pred, flat_label, average=score_averaging)
    rec = recall_score(flat_pred, flat_label, average=score_averaging)
    f1 = f1_score(flat_pred, flat_label, average=score_averaging)

    iou = compute_mean_iou(flat_pred, flat_label)

    return global_accuracy, class_accuracies, prec, rec, f1, iou

    
def compute_class_weights(labels_dir, label_values):
    '''
    Arguments:
        labels_dir(list): Directory where the image segmentation labels are
        num_classes(int): the number of classes of pixels in all images

    Returns:
        class_weights(list): a list of class weights where each index represents each class label and the element is the class weight for that label.

    '''
    image_files = [os.path.join(labels_dir, file) for file in os.listdir(labels_dir) if file.endswith('.png')]

    num_classes = len(label_values)

    class_pixels = np.zeros(num_classes) 

    total_pixels = 0.0

    for n in range(len(image_files)):
        image = imread(image_files[n])

        for index, colour in enumerate(label_values):
            class_map = np.all(np.equal(image, colour), axis = -1)
            class_map = class_map.astype(np.float32)
            class_pixels[index] += np.sum(class_map)

            
        print("\rProcessing image: " + str(n) + " / " + str(len(image_files)), end="")
        sys.stdout.flush()

    total_pixels = float(np.sum(class_pixels))
    index_to_delete = np.argwhere(class_pixels==0.0)
    class_pixels = np.delete(class_pixels, index_to_delete)

    class_weights = total_pixels / class_pixels
    class_weights = class_weights / np.sum(class_weights)

    return class_weights

# Compute the memory usage, for debugging
def memory():
    import os
    import psutil
    pid = os.getpid()
    py = psutil.Process(pid)
    memoryUse = py.memory_info()[0]/2.**30  # Memory use in GB
    print('Memory usage in GBs:', memoryUse)
