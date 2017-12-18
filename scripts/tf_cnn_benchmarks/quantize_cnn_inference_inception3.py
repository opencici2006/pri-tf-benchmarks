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

"""Benchmark script for TensorFlow.

See the README for more information.
"""

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from collections import defaultdict
import os
import threading
import time
import numpy as np
import tensorflow as tf
from tensorflow.python.client import timeline
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.platform import gfile
from google.protobuf import text_format

import cnn_util
import datasets
import model_config
import preprocessing

from setenvs import setenvs
from setenvs import arglist
import sys
args=arglist()


tf.flags.DEFINE_string('model_file', '/nfs/site/home/mbhuiyan/tensorflow/int8/int8_inception3_test/quantized_graph.pb',
                       """The quantized model file.""")

tf.flags.DEFINE_boolean('input_binary', False,
                       """The input model file is in binary format or not.""")

tf.flags.DEFINE_string('image_size', None,
                       """The image size.""")

tf.flags.DEFINE_integer('batch_size', 1, 'batch size per compute device')
tf.flags.DEFINE_integer('num_batches', 10,
                        'number of batches to run, excluding warmup')
tf.flags.DEFINE_integer('num_warmup_batches', None,
                        'number of batches to run before timing')

tf.flags.DEFINE_string('data_dir', '/dataset/sfu2/imagenet-data', """Path to dataset in TFRecord format
                       (aka Example protobufs). If not specified,
                       synthetic data will be used.""")
tf.flags.DEFINE_string('data_name', 'imagenet',
                       """Name of dataset: imagenet or flowers.
                       If not specified, it is automatically guessed
                       based on --data_dir.""")
tf.flags.DEFINE_string('resize_method', 'bilinear',
                       """Method for resizing input images:
                       crop,nearest,bilinear,bicubic or area.
                       The 'crop' mode requires source images to be at least
                       as large as the network input size,
                       while the other modes support any sizes and apply
                       random bbox distortions
                       before resizing (even with --nodistortions).""")
tf.flags.DEFINE_boolean('distortions', False,
                        """Enable/disable distortions during
                       image preprocessing. These include bbox and color
                       distortions.""")

tf.flags.DEFINE_string('data_format', 'NCHW',
                       """Data layout to use: NHWC (TF native)
                       or NCHW (cuDNN native).""")
tf.flags.DEFINE_integer('num_intra_threads', 1,
                        """Number of threads to use for intra-op
                       parallelism. If set to 0, the system will pick
                       an appropriate number.""")
tf.flags.DEFINE_integer('num_inter_threads', 0,
                        """Number of threads to use for inter-op
                       parallelism. If set to 0, the system will pick
                       an appropriate number.""")

tf.flags.DEFINE_string('timeline', 'quantized_timeline.json',
                       """The file name for time line.""")

tf.flags.DEFINE_string('tensorboard', 'tensorboard',
                       """The direcotry for tensorboard.""")

FLAGS = tf.flags.FLAGS
#FLAGS(sys.argv)

log_fn = print   # tf.logging.info

def get_image_and_label(dataset, input_nchan, image_size, batch_size,
                        input_data_type, resize_method):
  nclass = 1001

  """Add image Preprocessing ops to tf graph."""
  if dataset is not None:
    preproc_train = preprocessing.ImagePreprocessor(
        image_size, image_size, batch_size,
        1, input_data_type, True,
        resize_method=resize_method)

    images, labels = preproc_train.minibatch(dataset, subset='train')
    images = images[0]
  else:
    input_shape = [batch_size, image_size, image_size, input_nchan]
    images = tf.truncated_normal(
        input_shape,
        dtype=input_data_type,
        stddev=1e-1,
        name='synthetic_images')
    labels = tf.random_uniform(
        [batch_size],
        minval=1,
        maxval=nclass,
        dtype=tf.int32,
        name='synthetic_labels')

  return nclass, images, labels


def create_config_proto():
  config = tf.ConfigProto()
  config.allow_soft_placement = True
  config.intra_op_parallelism_threads = FLAGS.num_intra_threads
  config.inter_op_parallelism_threads = FLAGS.num_inter_threads
  config.gpu_options.force_gpu_compatible = FLAGS.force_gpu_compatible
  return config


def get_perf_timing_str(batch_size, step_train_times, scale=1):
  times = np.array(step_train_times)
  speeds = batch_size / times
  speed_mean = scale * batch_size / np.mean(times)
  if scale == 1:
    speed_uncertainty = np.std(speeds) / np.sqrt(float(len(speeds)))
    speed_madstd = 1.4826 * np.median(np.abs(speeds - np.median(speeds)))
    speed_jitter = speed_madstd
    return 'images/sec: %.1f +/- %.1f (jitter = %.1f)' % (
        speed_mean, speed_uncertainty, speed_jitter)
  else:
    return 'images/sec: %.1f' % speed_mean

def load_graph(model_file):
  graph = tf.Graph()
  graph_def = tf.GraphDef()

  with open(model_file, "rb") as f:
    if FLAGS.input_binary:
      graph_def.ParseFromString(f.read())
    else:
      text_format.Merge(f.read(), graph_def)

  with graph.as_default():
    tf.import_graph_def(graph_def)

  return graph

def read_tensor_from_image_file(file_name, input_height=299, input_width=299,
				input_mean=0, input_std=255):
  input_name = "file_reader"
  output_name = "normalized"
  file_reader = tf.read_file(file_name, input_name)
  if file_name.endswith(".png"):
    image_reader = tf.image.decode_png(file_reader, channels = 3,
                                       name='png_reader')
  elif file_name.endswith(".gif"):
    image_reader = tf.squeeze(tf.image.decode_gif(file_reader,
                                                  name='gif_reader'))
  elif file_name.endswith(".bmp"):
    image_reader = tf.image.decode_bmp(file_reader, name='bmp_reader')
  else:
    image_reader = tf.image.decode_jpeg(file_reader, channels = 3,
                                        name='jpeg_reader')
  float_caster = tf.cast(image_reader, tf.float32)
  dims_expander = tf.expand_dims(float_caster, 0);
  resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
  normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
  sess = tf.Session()
  result = sess.run(normalized)

  return result

def load_labels(label_file):
  label = []
  proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
  for l in proto_as_ascii_lines:
    label.append(l.rstrip())
  return label

if __name__ == "__main__":

  dataset = None
  data_name = FLAGS.data_name
  if FLAGS.data_dir is not None:
    if data_name is None:
      if 'imagenet' in FLAGS.data_dir:
        data_name = 'imagenet'
      elif 'flowers' in FLAGS.data_dir:
        data_name = 'flowers'
      else:
        raise ValueError('Could not identify name of dataset. '
                           'Please specify with --data_name option.')
    if data_name == 'imagenet':
      dataset = datasets.ImagenetData(FLAGS.data_dir)
    elif self.data_name == 'flowers':
      dataset = datasets.FlowersData(FLAGS.data_dir)
    else:
      raise ValueError('Unknown dataset. Must be one of imagenet or flowers.')

  image_size = 299
  input_layer = "input"
  output_layer = "InceptionV3/Predictions/Reshape_1"

  parser = argparse.ArgumentParser()
  parser.add_argument("--model_file", type=int, help="model size")
  parser.add_argument("--image_size", type=int, help="image size")
  parser.add_argument("--batch_size", type=int, help="batch size")
  parser.add_argument("--input_layer", help="name of input layer")
  parser.add_argument("--output_layer", help="name of output layer")
  args = parser.parse_args()

  model_file = FLAGS.model_file
  if args.model_file:
    model_file = args.model_file
  if model_file is None:
    raise ValueError('Please specify the quantized model file.')

  if args.image_size:
    image_size = args.image_size


  batch_size = FLAGS.batch_size
  if args.batch_size:
    batch_size = args.batch_size

  if args.input_layer:
    input_layer = args.input_layer
  if args.output_layer:
    output_layer = args.output_layer

  print( "Loading the quantized model" )

  graph = load_graph(model_file)

  input_name = "import/" + input_layer
  output_name = "import/" + output_layer
  input_operation = graph.get_operation_by_name(input_name);
  output_operation = graph.get_operation_by_name(output_name);

  print ("Get image data")
  nclass, images, labels = get_image_and_label(dataset, 3, image_size, batch_size,
                        tf.float32, FLAGS.resize_method)

  sess = tf.Session()
  result = sess.run(images)

  options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
  run_metadata = tf.RunMetadata()

  print("start inference")
  with tf.Session(graph=graph) as sess:
    for step in xrange(FLAGS.num_batches):
      print("Step: ", step)

      if FLAGS.timeline:
        results = sess.run(output_operation.outputs[0],
                        {input_operation.outputs[0]: result},
                         options = options, run_metadata = run_metadata
                        )
      else:
        results = sess.run(output_operation.outputs[0],
                        {input_operation.outputs[0]: result})

    if FLAGS.tensorboard:
      file_writer = tf.summary.FileWriter(FLAGS.tensorboard + '/graph',
                                      sess.graph)
  if FLAGS.timeline:
    fetched_timeline = timeline.Timeline(run_metadata.step_stats)
    chrome_trace = fetched_timeline.generate_chrome_trace_format()
    with open(FLAGS.timeline, 'w') as f:
      f.write(chrome_trace)

