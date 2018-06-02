import os
import sys
import numpy as np
import tensorflow as tf

slim = tf.contrib.slim
import time
import json
import davis
import logging
import matplotlib.pyplot as plt
from model.dataset import load_data, _read_py_function
from model.segmentation_model import model_init_fn, optimizer_init_fn, model_dim_print
from utils.util import load_image_files, check_image_dimension
import tensorflow.contrib.eager as tfe

from davis import *

env = "jingle.jiang"
train_model = True

if env == "jingle.jiang":
  tf.app.flags.DEFINE_string("root_path",
                           "/Users/jingle.jiang/personal/class/stanford/cs231n/final/video_segmentation/davis-2017/data/DAVIS",
                           "Available modes: train / show_examples / official_eval")
  tf.app.flags.DEFINE_string("output_path",
                             "/Users/jingle.jiang/personal/class/stanford/cs231n/final/video_segmentation/davis-2017/data/",
                             "output_path")
  tf.app.flags.DEFINE_string("device", "/cpu:0", "device")

elif env == "hyuna915":
  tf.app.flags.DEFINE_string("root_path",
                           "/Users/hyuna915/Desktop/2018-CS231N/Final_Project/video_segmentation/davis-2017/data/DAVIS",
                           "Available modes: train / show_examples / official_eval")
  tf.app.flags.DEFINE_string("device", "/cpu:0", "device")
  tf.app.flags.DEFINE_string("output_path",
                             "/Users/hyuna915/Desktop/2018-CS231N/Final_Project/video_segmentation/davis-2017/data/",
                             "output_path")

tf.app.flags.DEFINE_string("train_sequence", "elephant", "which sequence")
tf.app.flags.DEFINE_string("test_sequence", "elephant", "which sequence")

tf.app.flags.DEFINE_string("maskrcnn_label_path", "MaskRCNN/480p", "")
tf.app.flags.DEFINE_string("osvos_label_path", "Results/Segmentations/480p/OSVOS", "")
tf.app.flags.DEFINE_string("groundtruth_label_path", "Annotations/480p", "groundtruth_label_path")
tf.app.flags.DEFINE_string("groundtruth_image_path", "JPEGImages/480p", "groundtruth_image_path")
tf.app.flags.DEFINE_integer("height", 480, "height")
tf.app.flags.DEFINE_integer("weight", 854, "weight")
tf.app.flags.DEFINE_integer("num_epochs", 5, "num_epochs")

tf.app.flags.DEFINE_integer("filter", 64, "weight")
tf.app.flags.DEFINE_integer("kernel", 3, "weight")
tf.app.flags.DEFINE_integer("pad", 1, "weight")
tf.app.flags.DEFINE_integer("pool", 2, "weight")
tf.app.flags.DEFINE_integer("batch_size", 10, "batch_size")
tf.app.flags.DEFINE_float("lr", 0.00008, "learning rate")
tf.app.flags.DEFINE_boolean("layer32", True, "layer32")
tf.app.flags.DEFINE_boolean("layer64", True, "layer64")
tf.app.flags.DEFINE_boolean("layer128", True, "layer128")
tf.app.flags.DEFINE_boolean("layer256", True, "layer256")

tf.app.flags.DEFINE_string("test_mask_output",
                           "/Users/jingle.jiang/personal/class/stanford/cs231n/final/video_segmentation/davis-2017/data/DAVIS/unet",
                           "where to save test mask")
# tf.app.flags.DEFINE_string("device", "/gpu:0", "device")

FLAGS = tf.app.flags.FLAGS

file_handler = logging.FileHandler(os.path.join(FLAGS.output_path, "log.txt"))
console_handler = logging.StreamHandler()
logging.basicConfig(level=logging.INFO, handlers=[file_handler, console_handler])

def main(unused_argv):
  # Sanity check image dimension
  check_image_dimension(FLAGS)

  # construct image files array for training
  osvos_label_paths, maskrcnn_label_paths, groundtruth_label_paths = load_image_files(FLAGS, True)
  # construct image files array for training
  osvos_label_paths_test, maskrcnn_label_paths_test, groundtruth_label_paths_test = load_image_files(FLAGS, False)

  # import pdb; pdb.set_trace()

  # Generate tf.data.Dataset
  segmentation_dataset = load_data(osvos_label_paths, maskrcnn_label_paths, groundtruth_label_paths).batch(1)
  # for testing, the batch size should be all
  segmentation_dataset_test = load_data(osvos_label_paths_test,
                                        maskrcnn_label_paths_test,
                                        groundtruth_label_paths_test).batch(len(osvos_label_paths_test))

  # Get iterator on dataset
  # dataset_iterator = segmentation_dataset.make_one_shot_iterator()

  # successfully load dataset
  segmentation_dataset = load_data(osvos_label_paths, maskrcnn_label_paths, groundtruth_label_paths)
  segmentation_dataset = segmentation_dataset.shuffle(buffer_size=10000)
  segmentation_dataset = segmentation_dataset.batch(FLAGS.batch_size)
  # segmentation_dataset = segmentation_dataset.repeat(FLAGS.num_epochs)

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
    sess.run(tf.global_variables_initializer())
    logging.info("Global number of params: {}".format(sum(v.get_shape().num_elements() for v in tf.trainable_variables())))
    for epoch in range(FLAGS.num_epochs):
      logging.info("======================== Starting Epoch {} ========================".format(epoch))
      dataset_iterator = segmentation_dataset.make_one_shot_iterator()
      batch_num = 0
      while True:
        try:
          tic = time.time()
          batch= sess.run(dataset_iterator.get_next())
          x_np, y_np = batch
          logging.debug("x_list type {}, len(x_list)".format(type(x_np), len(y_np)))
          feed_dict = {x: x_np, y: y_np}
          loss_np, _ = sess.run([loss, train_op], feed_dict=feed_dict)
          toc = time.time()
          logging.info("Batch: {} Train Loss: {}, takes {} seconds".format(batch_num, loss_np, str(toc-tic)))
          batch_num += 1
        except tf.errors.OutOfRangeError:
          logging.warn("End of range")
          break

      # evaluate the model on train-val
      dataset_iterator_test = segmentation_dataset_test.make_one_shot_iterator()
      x_np, y_np = sess.run(dataset_iterator_test.get_next())
      feed_dict = {x: x_np, y: y_np}
      loss_test, pred_test = sess.run([loss, pred_mask], feed_dict=feed_dict)

      # import pdb; pdb.set_trace()
      mask_output_dir = "{}/{}/{}".format(FLAGS.test_mask_output, FLAGS.test_sequence, str(epoch))
      if not os.path.exists(mask_output_dir):
        os.makedirs(mask_output_dir)
      N_, H_, W_, _ = pred_test.shape
      # save predicted mask to somewhere
      for i in range(N_):
        mask_output = "{}/{:05d}.png".format(mask_output_dir, i)
        base_image = np.zeros((H_, W_))
        base_image[np.squeeze(pred_test[i,:,:,0]) > 0.5] = 1
        base_image[np.squeeze(pred_test[i, :, :, 0]) > 1.5] = 2
        base_image = base_image.astype(np.uint8)
        io.imwrite_indexed(mask_output, base_image)

    tf.train.Saver().save(sess=sess, save_path=FLAGS.output_path + "model.ckpt")

if __name__ == "__main__":
  tf.app.run()
