# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Generic evaluation script that evaluates a model using a given dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf

from datasets import dataset_factory
from nets import nets_factory
from preprocessing import preprocessing_factory

slim = tf.contrib.slim
boxe43 = [[0, 640, 0, 640], [320, 960, 0, 640], [640, 1280, 0, 640], [960, 1600, 0, 640], [1280, 1920, 0, 640], [0, 640, 320, 960], [320, 960, 320, 960], [640, 1280, 320, 960], [960, 1600, 320, 960], [1280, 1920, 320, 960], [0, 640, 640, 1280], [320, 960, 640, 1280], [640, 1280, 640, 1280], [960, 1600, 640, 1280], [1280, 1920, 640, 1280], [0, 640, 960, 1600], [320, 960, 960, 1600], [640, 1280, 960, 1600], [960, 1600, 960, 1600], [1280, 1920, 960, 1600], [0, 640, 1280, 1920], [320, 960, 1280, 1920], [640, 1280, 1280, 1920], [960, 1600, 1280, 1920], [1280, 1920, 1280, 1920], [0, 640, 1600, 2240], [320, 960, 1600, 2240], [640, 1280, 1600, 2240], [960, 1600, 1600, 2240], [1280, 1920, 1600, 2240], [0, 640, 1920, 2560], [320, 960, 1920, 2560], [640, 1280, 1920, 2560], [960, 1600, 1920, 2560], [1280, 1920, 1920, 2560]]
tf.app.flags.DEFINE_integer(
    'batch_size', 100, 'The number of samples in each batch.')

tf.app.flags.DEFINE_integer(
    'max_num_batches', None,
    'Max number of batches to evaluate by default use all.')

tf.app.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

tf.app.flags.DEFINE_string(
    'checkpoint_path', '/tmp/tfmodel/',
    'The directory where the model was written to or an absolute path to a '
    'checkpoint file.')

tf.app.flags.DEFINE_string(
    'eval_dir', '/tmp/tfmodel/', 'Directory where the results are saved to.')

tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 4,
    'The number of threads used to create the batches.')

tf.app.flags.DEFINE_string(
    'dataset_name', 'imagenet', 'The name of the dataset to load.')

tf.app.flags.DEFINE_string(
    'dataset_split_name', 'test', 'The name of the train/test split.')

tf.app.flags.DEFINE_string(
    'dataset_dir', None, 'The directory where the dataset files are stored.')

tf.app.flags.DEFINE_integer(
    'labels_offset', 0,
    'An offset for the labels in the dataset. This flag is primarily used to '
    'evaluate the VGG and ResNet architectures which do not use a background '
    'class for the ImageNet dataset.')

tf.app.flags.DEFINE_string(
    'model_name', 'inception_v3', 'The name of the architecture to evaluate.')

tf.app.flags.DEFINE_string(
    'preprocessing_name', None, 'The name of the preprocessing to use. If left '
    'as `None`, then the model_name flag is used.')

tf.app.flags.DEFINE_float(
    'moving_average_decay', None,
    'The decay to use for the moving average.'
    'If left as None, then moving averages are not used.')

tf.app.flags.DEFINE_integer(
    'eval_image_size', None, 'Eval image size')

FLAGS = tf.app.flags.FLAGS

def test_process(image, height, width):
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    #image = tf.to_float(image)
    #image = image / 255.0
    #image = tf.Print(image,[tf.reduce_max(image)])
    #image = tf.image.central_crop(image, central_fraction=central_fraction)
    image = tf.image.resize_images(image, [height, width], method=2, align_corners=False)
    image = tf.subtract(image, 0.5)
    image = tf.multiply(image, 2.0)
    return image
def main(_):
  from tensorflow.python.framework import graph_util
  output_graph_path = 'code/inv4_640.pb'
  with tf.Session() as sess:
    tf.global_variables_initializer().run()
    output_graph_def = tf.GraphDef()
    with open(output_graph_path, "rb") as f:
      output_graph_def.ParseFromString(f.read())
      _ = tf.import_graph_def(output_graph_def, name="")
    imagestr = sess.graph.get_tensor_by_name("input:0")
    score = sess.graph.get_tensor_by_name("score:0")
    ##############################################################
    ####################
    # Define the model #
    ####################
    import glob
    import cv2
    import numpy as np
    import os
    import datetime
    testPathes = glob.glob('./data/xuelang_round1_test_b/*jpg')
    li = ['filename,probability']
    sumpath = 0
    for path in testPathes:
        #img = cv2.imread(path)
        #img = img.astype(np.int32)
        #imgs = [tf.image.crop_to_bounding_box(img, box[0], box[2], 640, 640) for box in boxe43]
        image_data = tf.gfile.FastGFile(path, 'rb').read()
        ss = sess.run([score], feed_dict={imagestr:image_data})
        ss = np.array(ss).transpose()[0]
        ss.sort()
        max_ss = min(np.max(ss),0.99999)
        singg = os.path.basename(path)+','+"%.5f"%max_ss
        li.append(singg)
        print(singg)
        if max_ss>0.5:sumpath+=1
    line = '\n'.join(li)
    with open("./submit/submit_"+datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + ".csv",'w') as f:
        f.write(line)
    print(sumpath, len(testPathes))



if __name__ == '__main__':
  tf.app.run()

