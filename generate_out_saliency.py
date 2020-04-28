import os, time, cv2, sys, math, random
import tensorflow as tf
import argparse
import numpy as np

from utils import utils, helpers
from builders import model_builder

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', type=str, default='sal_75_sigmoid', required=False, help='The path to the latest checkpoint weights for your model.')
parser.add_argument('--crop_height', type=int, default=512, help='Height of cropped input image to network')
parser.add_argument('--crop_width', type=int, default=512, help='Width of cropped input image to network')
parser.add_argument('--model', type=str, default="Regression", required=False, help='The model you are using')
parser.add_argument('--dataset', type=str, default="CamVid", required=False, help='The dataset you are using')
args = parser.parse_args()

num_classes = 1

# Initializing network
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

net_input = tf.placeholder(tf.float32,shape=[None,None,None,3])
net_output = tf.placeholder(tf.float32,shape=[None,None,None,num_classes]) 

network, _ = model_builder.build_model(args.model, net_input = net_input, num_classes = num_classes, crop_width = args.crop_width, crop_height = args.crop_height, is_training = False)

sess.run(tf.global_variables_initializer())

# checkpoint_path = "%s/%04d/model.ckpt"%(args.checkpoint_path, 150)
checkpoint_path = '/home/mtech0/18CS60R31/AD/Semantic-Segmentation-Suite/sal_75_relu_50/latest_model_Regression_CamVid.ckpt'
output_path = "output_relu_e48"

print('Loading model checkpoint weights ...')
saver=tf.train.Saver(max_to_keep=1000)
saver.restore(sess, checkpoint_path)

# Load the data
print("Loading the data ...")
train_input_names, train_output_names, val_input_names, val_output_names, test_input_names, test_output_names = utils.prepare_data(dataset_dir=args.dataset)

# Create directories if needed
if not os.path.isdir("%s"%(output_path)):
    os.makedirs("%s"%(output_path))
    os.makedirs("%s/test_labels"%(output_path))
    os.makedirs("%s/test"%(output_path))
    os.makedirs("%s/val"%(output_path))
    os.makedirs("%s/val_labels"%(output_path))
    os.makedirs("%s/train"%(output_path))
    os.makedirs("%s/train_labels"%(output_path))

def get_output(img, label, threshold = 0.02):
    out = np.zeros(img.shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if label[i][j]> threshold:
                out[i][j] = img[i][j]
    return out

# Run testing on ALL test images
for ind in range(len(val_input_names)):
    sys.stdout.write("\rRunning val image %d / %d"%(ind+1, len(val_output_names)))
    sys.stdout.flush()

    file_name = utils.filepath_to_name(val_input_names[ind])
    file_name = os.path.basename(val_input_names[ind])
    file_name = os.path.splitext(file_name)[0]

    path = '../EyeTrackingData/CamVid/all_label_mix/'

    img = utils.load_image(val_input_names[ind])
    gt = utils.load_image(path + file_name+'.png')

    input_image, gt = utils.random_crop(img, gt, args.crop_height, args.crop_width)
    input_image = np.expand_dims(np.float32(input_image), axis = 0)/255.0
    output_image = sess.run(network,feed_dict={net_input:input_image})

    # out = cv2.cvtColor(output_image[0], cv2.COLOR_GRAY2RGB)*255
    # usable_output = cv2.multiply(input_image[0], out)*255

    usable_output = get_output(input_image[0]*255.0, output_image[0])

    cv2.imwrite("%s/val/%s.png"%(output_path, file_name),cv2.cvtColor(np.uint8(usable_output), cv2.COLOR_RGB2BGR))
    cv2.imwrite("%s/val_labels/%s.png"%(output_path, file_name),cv2.cvtColor(np.uint8(gt), cv2.COLOR_RGB2BGR))

for ind in range(len(test_input_names)):
    sys.stdout.write("\rRunning test image %d / %d"%(ind+1, len(test_output_names)))
    sys.stdout.flush()

    file_name = utils.filepath_to_name(test_input_names[ind])
    file_name = os.path.basename(test_input_names[ind])
    file_name = os.path.splitext(file_name)[0]

    path = '../EyeTrackingData/CamVid/all_label_mix/'

    img = utils.load_image(test_input_names[ind])
    gt = utils.load_image(path + file_name+'.png')

    input_image, gt = utils.random_crop(img, gt, args.crop_height, args.crop_width)
    input_image = np.expand_dims(np.float32(input_image), axis = 0)/255.0
    output_image = sess.run(network, feed_dict={net_input:input_image})

    # out = cv2.cvtColor(output_image[0], cv2.COLOR_GRAY2RGB)
    # usable_output = cv2.multiply(input_image[0], out)*255

    usable_output = get_output(input_image[0]*255.0, output_image[0])
    cv2.imwrite("%s/test/%s.png"%(output_path, file_name),cv2.cvtColor(np.uint8(usable_output), cv2.COLOR_RGB2BGR))
    cv2.imwrite("%s/test_labels/%s.png"%(output_path, file_name),cv2.cvtColor(np.uint8(gt), cv2.COLOR_RGB2BGR))

for ind in range(len(train_input_names)):
    sys.stdout.write("\rRunning test image %d / %d"%(ind+1, len(train_output_names)))
    sys.stdout.flush()

    file_name = utils.filepath_to_name(train_input_names[ind])
    file_name = os.path.basename(train_input_names[ind])
    file_name = os.path.splitext(file_name)[0]

    path = '../EyeTrackingData/CamVid/all_label_mix/'

    img = utils.load_image(train_input_names[ind])
    gt = utils.load_image(path + file_name+'.png')

    input_image, gt = utils.random_crop(img, gt, args.crop_height, args.crop_width)
    input_image = np.expand_dims(np.float32(input_image), axis = 0)/255.0
    output_image = sess.run(network, feed_dict={net_input:input_image})

    # out = cv2.cvtColor(output_image[0], cv2.COLOR_GRAY2RGB)
    # usable_output = cv2.multiply(input_image[0], out)*255
    
    usable_output = get_output(input_image[0]*255.0, output_image[0])
    cv2.imwrite("%s/train/%s.png"%(output_path, file_name),cv2.cvtColor(np.uint8(usable_output), cv2.COLOR_RGB2BGR))
    cv2.imwrite("%s/train_labels/%s.png"%(output_path, file_name),cv2.cvtColor(np.uint8(gt), cv2.COLOR_RGB2BGR))
