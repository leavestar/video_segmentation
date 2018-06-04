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
from model.dataset import load_data, load_data2, dimension_validation, get_channel_dim
from model.segmentation_model import model_init_fn, optimizer_init_fn, model_dim_print, dice_coefficient_loss
from utils.util import load_image_files, check_image_dimension, load_seq_from_yaml, path_config
import tensorflow.contrib.eager as tfe

from davis import *

env = "cloud"
path_config(env)

tf.app.flags.DEFINE_boolean("train_mode", True, "enable training")
tf.app.flags.DEFINE_boolean("debug_mode", False, "pdb debugger")
tf.app.flags.DEFINE_boolean("skip_test_mode", False, "skip test")


tf.app.flags.DEFINE_boolean("enable_osvos", False, "enable_maskrcnn")
tf.app.flags.DEFINE_boolean("enable_maskrcnn", False, "enable_maskrcnn")
tf.app.flags.DEFINE_boolean("enable_jpg", True, "enable_jpg")
tf.app.flags.DEFINE_boolean("enable_firstframe", True, "enable_firstframe")

tf.app.flags.DEFINE_string("maskrcnn_label_path", "MaskRCNN/480p", "maskrcnn_label_path")
tf.app.flags.DEFINE_string("osvos_label_path", "Results/Segmentations/480p/OSVOS2-convert", "osvos_label_path")
tf.app.flags.DEFINE_string("groundtruth_label_path", "Annotations/480p", "groundtruth_label_path")
tf.app.flags.DEFINE_string("groundtruth_image_path", "JPEGImages/480p", "groundtruth_image_path")

tf.app.flags.DEFINE_string("model_label", "", "model_label")

tf.app.flags.DEFINE_integer("height", 480, "height")
tf.app.flags.DEFINE_integer("weight", 854, "weight")
tf.app.flags.DEFINE_integer("filter", 64, "weight")
tf.app.flags.DEFINE_integer("kernel", 3, "weight")
tf.app.flags.DEFINE_integer("pad", 1, "weight")
tf.app.flags.DEFINE_integer("pool", 2, "weight")

tf.app.flags.DEFINE_integer("batch_size", 15, "batch_size")
tf.app.flags.DEFINE_integer("num_epochs", 1, "num_epochs")
tf.app.flags.DEFINE_float("lr", 0.00002, "learning rate")
tf.app.flags.DEFINE_integer("num_classes", 10, "num_classes")

tf.app.flags.DEFINE_boolean("layer32", True, "layer32")
tf.app.flags.DEFINE_boolean("layer64", True, "layer64")
tf.app.flags.DEFINE_boolean("layer128", True, "layer128")
tf.app.flags.DEFINE_boolean("layer256", False, "layer256")

tf.app.flags.DEFINE_integer("eval_every_n_epochs", 2, "eval on test every n trainig epoch")
tf.app.flags.DEFINE_integer("save_every_n_epochs", 5, "save on test every n trainig epoch")

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
    raise Exception("--model_label is required, eg osvos_maskrcnn_0602")

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
    train_seqs = load_seq_from_yaml(os.path.join(FLAGS.config_path, FLAGS.train_seq_yaml))

  train_val = load_seq_from_yaml(os.path.join(FLAGS.config_path, FLAGS.train_val_yaml))
  test_seqs = load_seq_from_yaml(os.path.join(FLAGS.config_path, FLAGS.test_val_yaml))

  return train_seqs, train_val, test_seqs


def generate_dataset(FLAGS, seqs, is_shuffle=False):
  osvos_label_paths, maskrcnn_label_paths, groundtruth_label_paths, groundtruth_image_paths, firstframe_image_paths = load_image_files(FLAGS, seqs)
  logger.info("Load {} image samples, sequences: {}".format(len(osvos_label_paths), seqs))

  if dimension_validation(
          osvos_label_paths,
          maskrcnn_label_paths,
          groundtruth_label_paths,
          groundtruth_image_paths,
          firstframe_image_paths,
          logger) is False:
    raise Exception("Invalid Image Found")

  # successfully load dataset
  segmentation_dataset, seqs = load_data2(FLAGS,
    osvos_label_paths,
    maskrcnn_label_paths,
    groundtruth_label_paths,
    groundtruth_image_paths,
    firstframe_image_paths)
  if is_shuffle:
    segmentation_dataset = segmentation_dataset.shuffle(buffer_size=10000)
  segmentation_dataset = segmentation_dataset.batch(FLAGS.batch_size)
  return segmentation_dataset, seqs


def main(unused_argv):
  train_seqs, train_sample, test_seqs = setup(root_path)
  # check_image_dimension(FLAGS, logger, train_seqs)

  # construct image files array
  if FLAGS.train_mode:
    segmentation_dataset, _ = generate_dataset(FLAGS, train_seqs)

  segmentation_dataset_val, val_seq_list = generate_dataset(FLAGS, train_sample)
  segmentation_dataset_test, test_seq_list = generate_dataset(FLAGS, test_seqs, False)

  with tf.device(FLAGS.device):
    x = tf.placeholder(tf.float32, [None, FLAGS.height, FLAGS.weight, 5])
    y = tf.placeholder(tf.float32, [None, FLAGS.height, FLAGS.weight, 1])

    # pred_mask = model_init_fn(FLAGS=FLAGS, inputs=x)
    pred_mask = model_dim_print(FLAGS=FLAGS, channel_dim=5, inputs=x)
    # loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=pred_mask)
    # loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=pred_mask)
    loss = dice_coefficient_loss(labels=y, logits=pred_mask)
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
      if FLAGS.train_mode:
        logger.info("======================== Starting training Epoch {} ========================".format(epoch))
        dataset_iterator = segmentation_dataset.make_one_shot_iterator()
        batch_num = 0
        while True:
          try:
            tic = time.time()
            batch = sess.run(dataset_iterator.get_next())
            _, _, _, x_np, y_np = batch
            # I notice this is shape (4, 480, 854, 2) and (4, 480, 854, 1), expected?
            # further, FLAGS.batch_size==10
            logger.debug("x_np type {}, shape {}".format(type(x_np), x_np.shape))
            logger.debug("y_np type {}, shape {}".format(type(y_np), y_np.shape))
            max_label = np.max(y_np)
            if max_label >= FLAGS.num_classes:
              logger.info("WRONG! {} > num_classes".format(max_label))
              continue
            feed_dict = {x: x_np, y: y_np}
            loss_np, _, = sess.run([loss, train_op], feed_dict=feed_dict)
            toc = time.time()
            logger.info("Batch: %i Train Loss: %.4f, takes %.2f seconds" % (batch_num, loss_np, toc - tic))
            # logger.info("total loss shape {}, value {}".format(total_loss_.shape, str(total_loss_)))
            batch_num += 1
          except tf.errors.OutOfRangeError:
            logger.warn("End of range")
            break

        if epoch % FLAGS.save_every_n_epochs == 0:
          if not os.path.exists(os.path.join(root_path, "models/tmp/epoch_{}/model.ckpt".format(str(epoch)))):
            os.makedirs(os.path.join(root_path, "models/tmp/epoch_{}/model.ckpt".format(str(epoch))))
          tf.train.Saver().save(sess=sess, save_path=root_path + "/models/tmp/epoch_{}/model.ckpt".format(str(epoch)))


      ##### evaluate the model on train-val
      if epoch % FLAGS.eval_every_n_epochs == 0:
        for target in ["train-val", "test"]:
          if FLAGS.skip_test_mode and target == "test":
            continue
          seq_dataset, seq_list = segmentation_dataset_test, test_seq_list
          if target == "train-val":
            seq_dataset, seq_list = segmentation_dataset_val, val_seq_list

          logger.info("======================== Starting testing Epoch {} - {} ========================".format(epoch, target))
          test_loss, davis_j, davis_f, davis_j_mean, davis_f_mean, time_taken = \
            eval_on_test_data2(sess, seq_dataset, seq_list, ops=[loss, pred_mask],
                              placeholder=[x, y], epoch=epoch, FLAGS=FLAGS)
          unique_seq_name = set(seq_list)
          detailed_loss = ""
          for seq_name_ in unique_seq_name:
            detailed_loss += "{}: J {}, F {}; ".format(seq_name_,
                                                       str(davis_j[seq_name_]["mean"][0]),
                                                       str(davis_f[seq_name_]["mean"][0]))
          logger.info(
            "{} Loss after epoch {}: "
            "cross-entropy: {}, "
            "mean J: {}, mean F: {}, "
            "DETAILS: {}; "
            "takes {} seconds"
              .format(target, epoch, test_loss, str(davis_j_mean), str(davis_f_mean), detailed_loss, str(time_taken)))


def eval_on_test_data(sess, segmentation_dataset_test, test_seq_list, ops, placeholder, epoch, FLAGS):
  # as of V2 on generator, test_seq_list is no longer in use.
  # import pdb; pdb.set_trace()
  tic = time.time()
  [x, y] = placeholder
  dataset_iterator_test = segmentation_dataset_test.make_one_shot_iterator()
  test_loss, davis_j, davis_f, test_n = 0.0, {}, {}, 0
  seq_name_iter = iter(test_seq_list)

  test_mask_output = os.path.join(root_path, "unet")
  # save predicted mask to somewhere
  frame_number_by_seq_name = {}
  while True:
    try:
      batch = sess.run(dataset_iterator_test.get_next())
      seq_name, image_number, object_number, x_np, y_np = batch

       # it seems x_np has shape (2, 480, 854, 2) and y_np has shape (2, 480, 854, 1)
       # this is a little unintuitive as we expect it to be batch_size
      feed_dict = {x: x_np, y: y_np}
      test_loss_, pred_test = sess.run(ops, feed_dict=feed_dict)

      # import pdb; pdb.set_trace()
      N_, H_, W_, C_ = pred_test.shape
      logging.info("pred_test.shape=={},{},{},{}".format(str(N_), str(H_), str(W_), str(C_)))
      # #TODO
      # if np.sum(pred_test> 1) > 0:
      #   logging.info("pred_test has {} >0".format(np.sum(pred_test> 1)))
      for i in range(N_):
        seq_name = next(seq_name_iter)
        seq_number = frame_number_by_seq_name.get(seq_name, 0)
        frame_number_by_seq_name[seq_name] = seq_number + 1
        # import pdb; pdb.set_trace()
        mask_output_dir = "{}/{}/{}/{}".format(test_mask_output, seq_name, str(epoch), seq_name)
        # print mask_output_dir
        if not os.path.exists(mask_output_dir):
          os.makedirs(mask_output_dir)

        mask_output = "{}/{:05d}.png".format(mask_output_dir, seq_number)

        #  pred_test is now (N_, H_, W_, num_class), we convert it to (H_, W_)
        # upon observation, turns out pred_test usually have very large 0 prediction equally large as other prediction.
        # to prevent constant 0 prediction, we reverse the list
        pred_test_ = pred_test[i, :, :, ::-1]
        base_image = np.squeeze(np.argmax(pred_test_, axis=-1))
        base_image = -base_image + FLAGS.num_classes - 1
        # above 3 line equlivalent to base_image = np.squeeze(np.argmax(pred_test[i, :, :, :], axis=-1))

        base_image = base_image.astype(np.uint8)
        io.imwrite_indexed(mask_output, base_image)
        if len(np.unique(base_image)) == 1:
          logger.info("problem on predicted base_iamge. maybe all 0 {}".format(mask_output))
          if FLAGS.debug_mode:
            import pdb; pdb.set_trace()

      test_n += 1
      test_loss += test_loss_
    except tf.errors.OutOfRangeError:
      logger.warn("End of test range")
      break

  # not finish yet! eval davis performance
  unique_seq_name = set(test_seq_list)
  davis_j_mean, davis_f_mean = 0.0, 0.0
  for seq_name in unique_seq_name:
    mask_output_dir = "{}/{}/{}/{}".format(test_mask_output, seq_name, str(epoch), seq_name)
    sg = Segmentation(mask_output_dir, False)

    ground_truth_dir_ = "{}/{}/{}".format(FLAGS.read_path, FLAGS.groundtruth_label_path, seq_name)
    ground_truth = Segmentation(ground_truth_dir_, False)

    davis_j[seq_name] = db_eval_sequence(sg, ground_truth, measure="J", n_jobs=32)
    davis_f[seq_name] = db_eval_sequence(sg, ground_truth, measure="F", n_jobs=32)
    davis_j_mean += davis_j[seq_name]["mean"][0]
    davis_f_mean += davis_f[seq_name]["mean"][0]

  toc = time.time()
  if test_n == 0:
    test_n = 1
  return test_loss / test_n, davis_j, davis_f, \
         davis_j_mean / len(unique_seq_name), davis_f_mean / len(unique_seq_name), toc - tic


def eval_on_test_data2(sess, segmentation_dataset_test, test_seq_list, ops, placeholder, epoch, FLAGS):
  # as of V2 on generator, test_seq_list is no longer in use.
  # import pdb; pdb.set_trace()
  tic = time.time()
  [x, y] = placeholder
  dataset_iterator_test = segmentation_dataset_test.make_one_shot_iterator()
  test_loss, davis_j, davis_f, test_n = 0.0, {}, {}, 0

  test_mask_output = os.path.join(root_path, "unet")
  # save predicted mask to somewhere
  while True:
    try:
      batch = sess.run(dataset_iterator_test.get_next())
      seq_name_list, image_number_list, object_number_list, x_np, y_np = batch

      feed_dict = {x: x_np, y: y_np}
      test_loss_, pred_test = sess.run(ops, feed_dict=feed_dict)
      # pred_test is of shape batch * H * W * 1 for each one of the object (multiple rows for same image)

      N_, H_, W_, C_ = pred_test.shape
      logging.info("pred_test.shape=={},{},{},{}".format(str(N_), str(H_), str(W_), str(C_)))
      # #TODO
      # if np.sum(pred_test> 1) > 0:
      #   logging.info("pred_test has {} >0".format(np.sum(pred_test> 1)))
      for i in range(N_):
        seq_name = seq_name_list[i]
        seq_number = image_number_list[i]
        object_number = object_number_list[i]

        mask_output_dir = "{}/{}/{}/{}".format(test_mask_output, seq_name, str(epoch), seq_name)
        # print mask_output_dir
        if not os.path.exists(mask_output_dir):
          os.makedirs(mask_output_dir)

        mask_output = "{}/{:05d}.png".format(mask_output_dir, seq_number)

        #  pred_test is now (N_, H_, W_, num_class), we convert it to (H_, W_)
        # upon observation, turns out pred_test usually have very large 0 prediction equally large as other prediction.
        # to prevent constant 0 prediction, we reverse the list
        pred_test_ = pred_test[i, :, :, ::-1]
        base_image = np.squeeze(np.argmax(pred_test_, axis=-1))
        base_image = -base_image + FLAGS.num_classes - 1
        # above 3 line equlivalent to base_image = np.squeeze(np.argmax(pred_test[i, :, :, :], axis=-1))

        base_image = base_image.astype(np.uint8)
        io.imwrite_indexed(mask_output, base_image)
        if len(np.unique(base_image)) == 1:
          logger.info("problem on predicted base_iamge. maybe all 0 {}".format(mask_output))
          if FLAGS.debug_mode:
            import pdb; pdb.set_trace()

      test_n += 1
      test_loss += test_loss_
    except tf.errors.OutOfRangeError:
      logger.warn("End of test range")
      break

  # not finish yet! eval davis performance
  unique_seq_name = set(test_seq_list)
  davis_j_mean, davis_f_mean = 0.0, 0.0
  for seq_name in unique_seq_name:
    mask_output_dir = "{}/{}/{}/{}".format(test_mask_output, seq_name, str(epoch), seq_name)
    sg = Segmentation(mask_output_dir, False)

    ground_truth_dir_ = "{}/{}/{}".format(FLAGS.read_path, FLAGS.groundtruth_label_path, seq_name)
    ground_truth = Segmentation(ground_truth_dir_, False)

    davis_j[seq_name] = db_eval_sequence(sg, ground_truth, measure="J", n_jobs=32)
    davis_f[seq_name] = db_eval_sequence(sg, ground_truth, measure="F", n_jobs=32)
    davis_j_mean += davis_j[seq_name]["mean"][0]
    davis_f_mean += davis_f[seq_name]["mean"][0]

  toc = time.time()
  if test_n == 0:
    test_n = 1
  return test_loss / test_n, davis_j, davis_f, \
         davis_j_mean / len(unique_seq_name), davis_f_mean / len(unique_seq_name), toc - tic


if __name__ == "__main__":
  tf.app.run()
