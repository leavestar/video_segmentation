import os
import sys
import numpy as np
import tensorflow as tf

slim = tf.contrib.slim
import davis
import logging
import matplotlib.pyplot as plt
from model.dataset import load_data, _read_py_function
from model.segmentation_model import model_init_fn, optimizer_init_fn, model_dim_print
from utils.util import load_image_files, check_image_dimension
import tensorflow.contrib.eager as tfe

# User defined parameters
seq_name = "car-shadow"
# env = "hyuna915"
env = "hyuna915"

train_model = True

if env == "jingle.jiang":
  tf.app.flags.DEFINE_string("root_path",
                           "/Users/jingle.jiang/personal/class/stanford/cs231n/final/video_segmentation/davis-2017/data/DAVIS",
                           "Available modes: train / show_examples / official_eval")
elif env == "hyuna915":
  tf.app.flags.DEFINE_string("root_path",
                           "/Users/hyuna915/Desktop/2018-CS231N/Final_Project/video_segmentation/davis-2017/data/DAVIS",
                           "Available modes: train / show_examples / official_eval")
  tf.app.flags.DEFINE_string("device", "/cpu:0", "device")

tf.app.flags.DEFINE_string("sequence", "elephant", "which sequence")
tf.app.flags.DEFINE_string("maskrcnn_label_path", "MaskRCNN/480p", "")
tf.app.flags.DEFINE_string("osvos_label_path", "Results/Segmentations/480p/OSVOS", "")
tf.app.flags.DEFINE_string("groundtruth_label_path", "Annotations/480p", "groundtruth_label_path")
tf.app.flags.DEFINE_string("groundtruth_image_path", "JPEGImages/480p", "groundtruth_image_path")
tf.app.flags.DEFINE_string("output_path", "/Users/hyuna915/Desktop/2018-CS231N/Final_Project/video_segmentation/davis-2017/data/", "output_path")
tf.app.flags.DEFINE_integer("height", 480, "height")
tf.app.flags.DEFINE_integer("weight", 854, "weight")
tf.app.flags.DEFINE_integer("num_epochs", 1, "num_epochs")

tf.app.flags.DEFINE_integer("filter", 64, "weight")
tf.app.flags.DEFINE_integer("kernel", 3, "weight")
tf.app.flags.DEFINE_integer("pad", 1, "weight")
tf.app.flags.DEFINE_integer("pool", 2, "weight")
# tf.app.flags.DEFINE_string("device", "/gpu:0", "device")

max_training_iters = 500
FLAGS = tf.app.flags.FLAGS

logging.basicConfig(level=logging.INFO)
file_handler = logging.FileHandler(os.path.join(FLAGS.output_path, "log.txt"))

def main(unused_argv):

  # Sanity check image dimension
  check_image_dimension(FLAGS)
  # construct image files array
  osvos_label_paths, maskrcnn_label_paths, groundtruth_label_paths = load_image_files(FLAGS)
  # Generate tf.data.Dataset
  segmentation_dataset = load_data(osvos_label_paths, maskrcnn_label_paths, groundtruth_label_paths).batch(1)

  # Get iterator on dataset
  dataset_iterator = segmentation_dataset.make_one_shot_iterator()
  # dataset = segmentation_dataset.batch(1)

  # successfully load dataset
  with tf.device(FLAGS.device):
    x = tf.placeholder(tf.float32, [None, FLAGS.height, FLAGS.weight, 2])
    y = tf.placeholder(tf.float32, [None, FLAGS.height, FLAGS.weight, 1])

    # pred_mask = model_init_fn(FLAGS=FLAGS, inputs=x)
    pred_mask = model_dim_print(FLAGS=FLAGS, inputs=x)
    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=pred_mask)
    loss = tf.reduce_mean(loss)
    optimizer = optimizer_init_fn(FLAGS=FLAGS)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      train_op = optimizer.minimize(loss)


  with tf.Session() as sess:
    # print(sess.run(next_element))
    sess.run(tf.global_variables_initializer())
    for epoch in range(FLAGS.num_epochs):
      while True:
        try:
          x_np, y_np = sess.run(dataset_iterator.get_next())
          print "x_np shape: {}, y_np shape: {}".format(x_np.shape, y_np.shape)
          feed_dict = {x: x_np, y: y_np}
          loss_np, _ = sess.run([loss, train_op], feed_dict=feed_dict)
        except tf.errors.OutOfRangeError:
          break


if __name__ == "__main__":
  tf.app.run()