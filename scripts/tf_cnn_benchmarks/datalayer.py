from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time
import os
import threading
import sys
from tensorflow.python.client import timeline
from six.moves import xrange  
import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import random_ops

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('dataset_size', 10104,
                            """Batch size.""")
tf.app.flags.DEFINE_integer('batch_size', 256,
                            """Batch size.""")
tf.app.flags.DEFINE_string('data_dir', '/home/shengfu/novartis/forIntel/tf',
                           """Path to the the processed data, i.e. """
                           """TFRecord of Example protos.""")
tf.app.flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
tf.app.flags.DEFINE_string('data_format', 'NCHW',
                           """The data format, can be either NHWC or NCHW.
                           """)
tf.app.flags.DEFINE_integer('data_pool', 1,
                            """Number of threads to be used for prefetching data.""")
tf.app.flags.DEFINE_integer('max_steps', 22732,
                            """Max number of iterations to run the training.""")
tf.app.flags.DEFINE_string('summaries_dir', './log',
                            """Directory where summaries are stored.""")
tf.app.flags.DEFINE_integer('num_epochs', 1,
                          """Number of epochs to run training on.""")
tf.app.flags.DEFINE_float('num_epochs_per_decay', 10.0,
                          """Epochs after which learning rate decays.""")
tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.16,
                          """Learning rate decay factor.""")
tf.app.flags.DEFINE_integer('inter_op', 2,
                           """Inter Op Parallelism Threads.""")
tf.app.flags.DEFINE_integer('intra_op', 64,
                           """Intra Op Parallelism Threads.""")

tf.app.flags.DEFINE_integer('height', 462,
                           """height.""")

tf.app.flags.DEFINE_integer('width', 581,
                           """width.""")

os.environ["KMP_BLOCKTIME"] = "1"
os.environ["OMP_NUM_THREADS"] = "80"
#os.environ["KMP_AFFINITY"]= "granularity=fine,compact,1,0" 

# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Image pre-processing utilities.
"""
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from tensorflow.python.ops import data_flow_ops
import cnn_util

FLAGS = tf.flags.FLAGS


def parse_example_proto(example_serialized):
  """Parses an Example proto containing a training example of an image.

  The output of the build_image_data.py image preprocessing script is a dataset
  containing serialized Example protocol buffers. Each Example proto contains
  the following fields:

    image/height: 462
    image/width: 581
    image/colorspace: 'RGB'
    image/channels: 3
    image/class/label: 615
    image/class/synset: 'n03623198'
    image/class/text: 'knee pad'
    image/object/bbox/xmin: 0.1
    image/object/bbox/xmax: 0.9
    image/object/bbox/ymin: 0.2
    image/object/bbox/ymax: 0.6
    image/object/bbox/label: 615
    image/format: 'JPEG'
    image/filename: 'ILSVRC2012_val_00041207.JPEG'
    image/encoded: <JPEG encoded string>

  Args:
    example_serialized: scalar Tensor tf.string containing a serialized
      Example protocol buffer.

  Returns:
    image_buffer: Tensor tf.string containing the contents of a JPEG file.
    label: Tensor tf.int32 containing the label.
    bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
      where each coordinate is [0, 1) and the coordinates are arranged as
      [ymin, xmin, ymax, xmax].
    text: Tensor tf.string containing the human-readable label.
  """
  # Dense features in Example proto.
  feature_map = {
      'image/encoded': tf.FixedLenFeature([], dtype=tf.string,
                                          default_value=''),
      'image/class/label': tf.FixedLenFeature([1], dtype=tf.int64,
                                              default_value=-1),
      'image/class/text': tf.FixedLenFeature([], dtype=tf.string,
                                             default_value=''),
  }
  sparse_float32 = tf.VarLenFeature(dtype=tf.float32)
  # Sparse features in Example proto.
  feature_map.update(
      {k: sparse_float32 for k in ['image/object/bbox/xmin',
                                   'image/object/bbox/ymin',
                                   'image/object/bbox/xmax',
                                   'image/object/bbox/ymax']})

  features = tf.parse_single_example(example_serialized, feature_map)
  label = tf.cast(features['image/class/label'], dtype=tf.int32)

  xmin = tf.expand_dims(features['image/object/bbox/xmin'].values, 0)
  ymin = tf.expand_dims(features['image/object/bbox/ymin'].values, 0)
  xmax = tf.expand_dims(features['image/object/bbox/xmax'].values, 0)
  ymax = tf.expand_dims(features['image/object/bbox/ymax'].values, 0)

  # Note that we impose an ordering of (y, x) just to make life difficult.
  bbox = tf.concat([ymin, xmin, ymax, xmax], 0)

  # Force the variable number of bounding boxes into the shape
  # [1, num_boxes, coords].
  bbox = tf.expand_dims(bbox, 0)
  bbox = tf.transpose(bbox, [0, 2, 1])

  return features['image/encoded'], label, bbox, features['image/class/text']


def decode_jpeg(image_buffer):  # , dtype=tf.float32):
  """Decode a JPEG string into one 3-D float image Tensor.

  Args:
    image_buffer: scalar string Tensor.
  Returns:
    3-D float Tensor with values ranging from [0, 1).
  """
  with tf.name_scope('decode_jpeg'):
    # Decode the string as an RGB JPEG.
    # Note that the resulting image contains an unknown height and width
    # that is set dynamically by decode_jpeg. In other words, the height
    # and width of image is unknown at compile-time.
    image = tf.image.decode_jpeg(image_buffer, channels=3,
                                 fancy_upscaling=False,
                                 dct_method='INTEGER_FAST')

    # image = tf.Print(image, [tf.shape(image)], 'Image shape: ')

    return image


def eval_image(image, height, width, bbox, resize):
  """Get the image for model evaluation."""
  with tf.name_scope('eval_image'):
    if resize == 'crop':
      # Note: This is much slower than crop_to_bounding_box
      #         It seems that the redundant pad step has huge overhead
      # distorted_image = tf.image.resize_image_with_crop_or_pad(image,
      #                                                         height, width)
      shape = tf.shape(image)
      y0 = (shape[0] - height) // 2
      x0 = (shape[1] - width) // 2
      # distorted_image = tf.slice(image, [y0,x0,0], [height,width,3])
      distorted_image = tf.image.crop_to_bounding_box(image, y0, x0, height,
                                                      width)
    else:
      sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
          tf.shape(image),
          bounding_boxes=bbox,
          min_object_covered=0.1,
          aspect_ratio_range=[0.75, 1.33],
          area_range=[0.05, 1.0],
          max_attempts=100,
          use_image_if_no_bounding_boxes=True)
      bbox_begin, bbox_size, _ = sample_distorted_bounding_box
      # Crop the image to the specified bounding box.
      distorted_image = tf.slice(image, bbox_begin, bbox_size)
      resize_method = {
          'nearest': tf.image.ResizeMethod.NEAREST_NEIGHBOR,
          'bilinear': tf.image.ResizeMethod.BILINEAR,
          'bicubic': tf.image.ResizeMethod.BICUBIC,
          'area': tf.image.ResizeMethod.AREA
      }[resize]
      # This resizing operation may distort the images because the aspect
      # ratio is not respected.
      if cnn_util.tensorflow_version() >= 11:
        distorted_image = tf.image.resize_images(
            distorted_image, [height, width],
            resize_method,
            align_corners=False)
      else:
        distorted_image = tf.image.resize_images(
            distorted_image, height, width, resize_method, align_corners=False)
    distorted_image.set_shape([height, width, 3])
    image = distorted_image
  return image


def distort_image(image, height, width, bbox):
  """Distort one image for training a network.

  Distorting images provides a useful technique for augmenting the data
  set during training in order to make the network invariant to aspects
  of the image that do not effect the label.

  Args:
    image: 3-D float Tensor of image
    height: integer
    width: integer
    bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
      where each coordinate is [0, 1) and the coordinates are arranged
      as [ymin, xmin, ymax, xmax].
  Returns:
    3-D float Tensor of distorted image used for training.
  """
  # with tf.op_scope([image, height, width, bbox], scope, 'distort_image'):
  # with tf.name_scope(scope, 'distort_image', [image, height, width, bbox]):
  with tf.name_scope('distort_image'):
    # Each bounding box has shape [1, num_boxes, box coords] and
    # the coordinates are ordered [ymin, xmin, ymax, xmax].

    # After this point, all image pixels reside in [0,1)
    # until the very end, when they're rescaled to (-1, 1).  The various
    # adjust_* ops all require this range for dtype float.
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)

  # A large fraction of image datasets contain a human-annotated bounding
  # box delineating the region of the image containing the object of interest.
  # We choose to create a new bounding box for the object which is a randomly
  # distorted version of the human-annotated bounding box that obeys an allowed
  # range of aspect ratios, sizes and overlap with the human-annotated
  # bounding box. If no box is supplied, then we assume the bounding box is
  # the entire image.
    sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
        tf.shape(image),
        bounding_boxes=bbox,
        min_object_covered=0.1,
        aspect_ratio_range=[0.75, 1.33],
        area_range=[0.05, 1.0],
        max_attempts=100,
        use_image_if_no_bounding_boxes=True)
    bbox_begin, bbox_size, distort_bbox = sample_distorted_bounding_box

    # Crop the image to the specified bounding box.
    distorted_image = tf.slice(image, bbox_begin, bbox_size)

    # This resizing operation may distort the images because the aspect
    # ratio is not respected. We select a resize method in a round robin
    # fashion based on the thread number.
    # Note that ResizeMethod contains 4 enumerated resizing methods.
    # TODO: how to chnage resize_method
    resize_method = 0
    if cnn_util.tensorflow_version() >= 11:
      distorted_image = tf.image.resize_images(
          distorted_image, [height, width], resize_method, align_corners=False)
    else:
      distorted_image = tf.image.resize_images(
          distorted_image, height, width, resize_method, align_corners=False)
    # Restore the shape since the dynamic slice based upon the bbox_size loses
    # the third dimension.
    distorted_image.set_shape([height, width, 3])

    # Randomly flip the image horizontally.
    distorted_image = tf.image.random_flip_left_right(distorted_image)

    # Randomly distort the colors.
    distorted_image = distort_color(distorted_image)

    # Note: This ensures the scaling matches the output of eval_image
    distorted_image *= 256

    return distorted_image


def distort_color(image):
  """Distort the color of the image.

  Each color distortion is non-commutative and thus ordering of the color ops
  matters. Ideally we would randomly permute the ordering of the color ops.
  Rather then adding that level of complication, we select a distinct ordering
  of color ops for each preprocessing thread.

  Args:
    image: Tensor containing single image.
  Returns:
    color-distorted image
  """
  # with tf.op_scope([image], scope, 'distort_color'):
  # with tf.name_scope(scope, 'distort_color', [image]):
  with tf.name_scope('distort_color'):
    # TODO: how to change color_ordering
    color_ordering = 0

    if color_ordering == 0:
      image = tf.image.random_brightness(image, max_delta=32. / 255.)
      image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
      image = tf.image.random_hue(image, max_delta=0.2)
      image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
    elif color_ordering == 1:
      image = tf.image.random_brightness(image, max_delta=32. / 255.)
      image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
      image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
      image = tf.image.random_hue(image, max_delta=0.2)

    # The random_* ops do not necessarily clamp.
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image

def preprocess(image_buffer, label, bbox, text):
    # Note: Width and height of image is known only at runtime.
    image = tf.image.decode_jpeg(image_buffer, channels=3,
                                 dct_method='INTEGER_FAST')

    image = distort_image(image, FLAGS.height, FLAGS.width, bbox)

    return image, label, bbox, text

def run_datalayer():
	with tf.Graph().as_default():  

		data_file = ["/localdisk/sfu2/imagenet-data/train-00120-of-00128", 
                  	 "/localdisk/sfu2/imagenet-data/train-00121-of-00128",
                  	 "/localdisk/sfu2/imagenet-data/train-00122-of-00128", 
                  	 "/localdisk/sfu2/imagenet-data/train-00123-of-00128", 
                  	 "/localdisk/sfu2/imagenet-data/train-00124-of-00128", 
                  	 "/localdisk/sfu2/imagenet-data/train-00125-of-00128", 
                  	 "/localdisk/sfu2/imagenet-data/train-00126-of-00128",                   
                  	 "/localdisk/sfu2/imagenet-data/train-00127-of-00128"
		            ]
		#data_file = ["/SATAdisk/imagenet-data/train-00120-of-00128", 
                #  	 "/SATAdisk/imagenet-data/train-00121-of-00128",
                #  	 "/SATAdisk/imagenet-data/train-00122-of-00128", 
                #  	 "/SATAdisk/imagenet-data/train-00123-of-00128", 
                #  	 "/SATAdisk/imagenet-data/train-00124-of-00128", 
                #  	 "/SATAdisk/imagenet-data/train-00125-of-00128", 
                #  	 "/SATAdisk/imagenet-data/train-00126-of-00128",                   
                #  	 "/SATAdisk/imagenet-data/train-00127-of-00128"
		#            ]
		dataset = tf.contrib.data.TFRecordDataset(data_file)
		dataset = dataset.map(parse_example_proto, num_threads=16, output_buffer_size=80)
		dataset = dataset.map(preprocess, num_threads=16, output_buffer_size=80)
		batched_dataset = dataset.batch(FLAGS.batch_size)
		iterator = batched_dataset.make_one_shot_iterator()

		image, label, bbox, text = iterator.get_next()
		tf.summary.image('images', image, max_outputs=FLAGS.batch_size)	

		config = tf.ConfigProto(inter_op_parallelism_threads = FLAGS.inter_op,
                                intra_op_parallelism_threads = FLAGS.intra_op)

		#run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
		#run_metadata = tf.RunMetadata()

		sess = tf.Session(config=config)

		merged = tf.summary.merge_all()
		train_writer = tf.summary.FileWriter(FLAGS.summaries_dir, sess.graph)
		
        max_steps = 10 
        total_duration = 0.0
        for step in xrange(max_steps):

            start = time.time()
            #_, summary = sess.run([iterator.get_next(), merged])
            sess.run(iterator.get_next())
			
            duration = time.time() - start
            print('Step:%d , duration: %.3f s ' % (step, duration))
            total_duration += duration
            
            #train_writer.add_summary(summary)
			
		###TRACE STUFF ###
		# tl = timeline.Timeline(run_metadata.step_stats)
		# ctf = tl.generate_chrome_trace_format(show_memory=True)
		# with open('mcnn-training-timeline.json', 'w') as f:
		# 	f.write(ctf)
		###TRACE STUFF ###
        print('Average duration for each iteration = %f' %(total_duration / max_steps))	
        sess.close()
		 
def main(_):
	run_datalayer()


if __name__ == '__main__':
	tf.app.run()



    



