import os,time,cv2, sys, math
import tensorflow as tf
import argparse
import numpy as np

from utils import utils, helpers
from builders import model_builder

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', type=str, default='checkpoint_cameye2', required=False, help='The path to the latest checkpoint weights for your model.')
parser.add_argument('--crop_height', type=int, default=512, help='Height of cropped input image to network')
parser.add_argument('--crop_width', type=int, default=512, help='Width of cropped input image to network')
parser.add_argument('--model', type=str, default="PSPNet", required=False, help='The model you are using')
parser.add_argument('--dataset', type=str, default="CamEye2", required=False, help='The dataset you are using')
args = parser.parse_args()

# Get the names of the classes so we can record the evaluation results
print("Retrieving dataset information ...")
class_names_list, label_values = helpers.get_label_info(os.path.join(args.dataset, "class_dict.csv"))
class_names_string = ""
for class_name in class_names_list:
    if not class_name == class_names_list[-1]:
        class_names_string = class_names_string + class_name + ", "
    else:
        class_names_string = class_names_string + class_name

num_classes = len(label_values)

# Initializing network
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess=tf.Session(config=config)

net_input = tf.placeholder(tf.float32,shape=[None,None,None,3])
net_output = tf.placeholder(tf.float32,shape=[None,None,None,num_classes]) 

network, _ = model_builder.build_model(args.model, net_input=net_input, num_classes=num_classes, crop_width=args.crop_width, crop_height=args.crop_height, is_training=False)

sess.run(tf.global_variables_initializer())

checkpoint_path = "%s/%04d/model.ckpt"%(args.checkpoint_path, 186)
output_path = "Test"

print('Loading model checkpoint weights ...')
saver=tf.train.Saver(max_to_keep=1000)
saver.restore(sess, checkpoint_path)

# Load the data
print("Loading the data ...")
train_input_names,train_output_names, val_input_names, val_output_names, test_input_names, test_output_names = utils.prepare_data(dataset_dir=args.dataset)

# Create directories if needed
if not os.path.isdir("%s"%(output_path)):
    os.makedirs("%s"%(output_path))

# def randomcropX(img, size = (512, 512)):
#     working = img.copy()
#     x, y, z = img.shape
#     data1 = working[0:size[0], 0:size[1], :]/255.0
#     data2 = working[x-size[0]:x, 0:size[1], :]/255.0
#     data3 = working[x-size[0]:x, y-size[1]:y, :]/255.0
#     data4 = working[0:size[0]:, y-size[1]:y, :]/255.0
#     return np.array([data1, data2, data3, data4])

# Run testing on ALL test images
for ind in range(len(test_input_names)):
    sys.stdout.write("\rRunning test image %d / %d"%(ind+1, len(test_input_names)))
    sys.stdout.flush()

    input_image = np.float32(cv2.imread(test_input_names[ind]))
    # inputs = randomcropX(input_image)
    input_image = np.expand_dims(input_image, axis=0)

    outputs = sess.run(network,feed_dict={net_input:input_image})
    file_name = utils.filepath_to_name(val_input_names[ind])

    for i in range(len(outputs)):
        output_image = np.array(outputs[i,:,:,:])
        output_image = helpers.reverse_one_hot(output_image)
        out_vis_image = helpers.colour_code_segmentation(output_image, label_values)

        cv2.imwrite("%s/%s_%d_pred.png"%(output_path, file_name, i),cv2.cvtColor(np.uint8(out_vis_image), cv2.COLOR_RGB2BGR))

