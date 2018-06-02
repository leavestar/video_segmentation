import os
import sys
import numpy as np
import tensorflow as tf

slim = tf.contrib.slim
import time
import json
import yaml
import davis
import logging
import matplotlib.pyplot as plt
from model.dataset import load_data, _read_py_function
from model.segmentation_model import model_init_fn, optimizer_init_fn, model_dim_print
from utils.util import load_image_files, check_image_dimension
import tensorflow.contrib.eager as tfe

from davis import *

env = "hyuna915"
train_model = True

if env == "jingle.jiang":

  tf.app.flags.DEFINE_string("read_path",
                             "/Users/jingle.jiang/personal/class/stanford/cs231n/final/video_segmentation/davis-2017/data/DAVIS",
                             "Available modes: train / show_examples / official_eval")
elif env == "hyuna915":
  tf.app.flags.DEFINE_string("read_path",
                             "/Users/hyuna915/Desktop/2018-CS231N/Final_Project/video_segmentation/davis-2017/data/DAVIS",
                             "Available modes: train / show_examples / official_eval")
  tf.app.flags.DEFINE_string("device", "/cpu:0", "device")

tf.app.flags.DEFINE_boolean("train_mode", True, "enable training")

tf.app.flags.DEFINE_string("maskrcnn_label_path", "MaskRCNN/480p", "maskrcnn_label_path")
tf.app.flags.DEFINE_string("osvos_label_path", "Results/Segmentations/480p/OSVOS", "osvos_label_path")
tf.app.flags.DEFINE_string("groundtruth_label_path", "Annotations/480p", "groundtruth_label_path")
tf.app.flags.DEFINE_string("groundtruth_image_path", "JPEGImages/480p", "groundtruth_image_path")
tf.app.flags.DEFINE_string("output_path",
                           "/Users/hyuna915/Desktop/2018-CS231N/Final_Project/video_segmentation/davis-2017/data/",
                           "output_path")
tf.app.flags.DEFINE_string("model_label", "", "model_label")


tf.app.flags.DEFINE_integer("height", 480, "height")
tf.app.flags.DEFINE_integer("weight", 854, "weight")
tf.app.flags.DEFINE_integer("filter", 64, "weight")
tf.app.flags.DEFINE_integer("kernel", 3, "weight")
tf.app.flags.DEFINE_integer("pad", 1, "weight")
tf.app.flags.DEFINE_integer("pool", 2, "weight")

tf.app.flags.DEFINE_integer("batch_size", 10, "batch_size")
tf.app.flags.DEFINE_integer("num_epochs", 1, "num_epochs")
tf.app.flags.DEFINE_float("lr", 0.00003, "learning rate")

tf.app.flags.DEFINE_boolean("layer32", True, "layer32")
tf.app.flags.DEFINE_boolean("layer64", True, "layer64")
tf.app.flags.DEFINE_boolean("layer128", True, "layer128")
tf.app.flags.DEFINE_boolean("layer256", True, "layer256")

tf.app.flags.DEFINE_string("test_mask_output",
                           "/Users/jingle.jiang/personal/class/stanford/cs231n/final/video_segmentation/davis-2017/data/DAVIS/unet",
                           "where to save test mask")
tf.app.flags.DEFINE_integer("eval_every_n_epochs", 1, "eval on test every n trainig epoch")
tf.app.flags.DEFINE_integer("save_every_n_epochs", 1, "save on test every n trainig epoch")
# tf.app.flags.DEFINE_string("device", "/gpu:0", "device")

FLAGS = tf.app.flags.FLAGS

root_path = os.path.join(FLAGS.output_path, FLAGS.model_label)

if not os.path.exists(root_path):
  os.makedirs(root_path)

file_handler = logging.FileHandler(root_path + "/log.txt")
file_handler.setLevel(logging.INFO)
logger = logging.getLogger('server_logger')
logger.addHandler(file_handler)

def setup(root_path):
  if not FLAGS.model_label:
     raise Exception("--model_label is required")

  if not os.path.exists(root_path + "/models"):
    os.makedirs(root_path + "/models")
    logger.info("Output path not found, create {}".format(root_path + "/models"))
  else:
    logger.info("Output Path found {}".format(root_path + "/models"))

  with open(os.path.join(root_path, "flags.json"), 'w') as fout:
      json.dump(FLAGS.flag_values_dict(), fout)
  logger.info("Flags: {}".format(FLAGS.flag_values_dict()))

  train_seqs = None
  if FLAGS.train_mode:
    train_path = os.path.join('.', 'train.yaml')
    with open(train_path, 'r') as train_stream:
      train_dict = yaml.load(train_stream)
    train_seqs = train_dict['sequences']

  test_path = os.path.join('.','test.yaml')
  with open(test_path, 'r') as test_stream:
    test_dict = yaml.load(test_stream)
  test_seqs = test_dict['sequences']
  return train_seqs, test_seqs


def generate_dataset(FLAGS, seqs):
  osvos_label_paths, maskrcnn_label_paths, groundtruth_label_paths = load_image_files(FLAGS, seqs)
  logger.info("Load {} image samples, sequences: {}".format(len(osvos_label_paths), seqs))
  # successfully load dataset
  segmentation_dataset = load_data(osvos_label_paths, maskrcnn_label_paths, groundtruth_label_paths)
  segmentation_dataset = segmentation_dataset.shuffle(buffer_size=10000)
  segmentation_dataset = segmentation_dataset.batch(FLAGS.batch_size)
  return segmentation_dataset


def main(unused_argv):
  train_seqs, test_seqs = setup(root_path)
  check_image_dimension(FLAGS, logger, train_seqs)

  # construct image files array
  if FLAGS.train_mode:
    segmentation_dataset = generate_dataset(FLAGS, train_seqs)

  segmentation_dataset_test = generate_dataset(FLAGS, test_seqs)

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
    logger.info(
      "Global number of params: {}".format(sum(v.get_shape().num_elements() for v in tf.trainable_variables())))
    for epoch in range(FLAGS.num_epochs):
      logger.info("======================== Starting Epoch {} ========================".format(epoch))
      dataset_iterator = segmentation_dataset.make_one_shot_iterator()
      batch_num = 0
      while True:
        try:
          tic = time.time()
          batch = sess.run(dataset_iterator.get_next())
          x_np, y_np = batch
          logging.debug("x_list type {}, len(x_list)".format(type(x_np), len(y_np)))
          feed_dict = {x: x_np, y: y_np}
          loss_np, _ = sess.run([loss, train_op], feed_dict=feed_dict)
          toc = time.time()
          logger.info("Batch: %i Train Loss: %.4f, takes %.2f seconds" % (batch_num, loss_np, toc - tic))
          batch_num += 1
        except tf.errors.OutOfRangeError:
          logger.warn("End of range")
          break

      # evaluate the model on train-val
      if epoch % FLAGS.eval_every_n_epochs ==0:
        dataset_iterator_test = segmentation_dataset_test.make_one_shot_iterator()
        test_loss, davis_j, davis_f, test_n = 0.0, 0.0, 0.0, 0
        while True:
          try:
            x_np, y_np = sess.run(dataset_iterator_test.get_next())
            feed_dict = {x: x_np, y: y_np}
            test_loss_, pred_test = sess.run([loss, pred_mask], feed_dict=feed_dict)

            # import pdb; pdb.set_trace()
            mask_output_dir = "{}/{}/{}".format(FLAGS.test_mask_output, FLAGS.test_sequence, str(epoch))
            if not os.path.exists(mask_output_dir):
              os.makedirs(mask_output_dir)
            N_, H_, W_, _ = pred_test.shape
            # save predicted mask to somewhere
            for i in range(N_):
              mask_output = "{}/{:05d}.png".format(mask_output_dir, i+test_n)
              base_image = np.zeros((H_, W_))
              base_image[np.squeeze(pred_test[i,:,:,0]) > 0.5] = 1
              base_image[np.squeeze(pred_test[i, :, :, 0]) > 1.5] = 2
              base_image[np.squeeze(pred_test[i, :, :, 0]) > 2.5] = 3
              base_image[np.squeeze(pred_test[i, :, :, 0]) > 3.5] = 4
              base_image[np.squeeze(pred_test[i, :, :, 0]) > 4.5] = 5
              base_image[np.squeeze(pred_test[i, :, :, 0]) > 5.5] = 6
              base_image[np.squeeze(pred_test[i, :, :, 0]) > 6.5] = 7
              base_image = base_image.astype(np.uint8)
              io.imwrite_indexed(mask_output, base_image)

              # not finish yet! eval davis performance

            test_n += 1
            test_loss += test_loss_

          except tf.errors.OutOfRangeError:
            logging.warn("End of test range")
            break
        logging.info("Test Loss after batch {}: {}, takes {} seconds".format(batch_num, test_loss/test_n, str(toc - tic)))

    tf.train.Saver().save(sess=sess, save_path=root_path +"/models/model.ckpt")


if __name__ == "__main__":
  tf.app.run()
