import os
import re
import sys
import numpy as np
import tensorflow as tf
import davis


def load_image_files(FLAGS):
  osvos_label_path = "{}/{}/{}/".format(FLAGS.root_path, FLAGS.osvos_label_path, FLAGS.sequence)
  maskrcnn_label_path = "{}/{}/{}/".format(FLAGS.root_path, FLAGS.maskrcnn_label_path, FLAGS.sequence)
  groundtruth_label_path = "{}/{}/{}/".format(FLAGS.root_path, FLAGS.groundtruth_label_path, FLAGS.sequence)
  groundtruth_image_path = "{}/{}/{}/".format(FLAGS.root_path, FLAGS.groundtruth_image_path, FLAGS.sequence)

  osvos_label_paths = []
  maskrcnn_label_paths = []
  groundtruth_label_paths = []
  groundtruth_image_paths = []

  for file in os.listdir(groundtruth_label_path):
    if re.match(r'[0-9]+.*\.png', file):
      osvos_label_paths.append(osvos_label_path + file)
      maskrcnn_label_paths.append(maskrcnn_label_path + file)
      groundtruth_label_paths.append(groundtruth_label_path + file)
      print groundtruth_label_path + file
      print osvos_label_path + file
      print maskrcnn_label_path + file

  return osvos_label_paths, maskrcnn_label_paths, groundtruth_label_paths


def check_image_dimension(FLAGS):
  osvos_label_path = "{}/{}/{}/".format(FLAGS.root_path, FLAGS.osvos_label_path, FLAGS.sequence)
  maskrcnn_label_path = "{}/{}/{}/".format(FLAGS.root_path, FLAGS.maskrcnn_label_path, FLAGS.sequence)
  groundtruth_label_path = "{}/{}/{}/".format(FLAGS.root_path, FLAGS.groundtruth_label_path, FLAGS.sequence)
  groundtruth_image_path = "{}/{}/{}/".format(FLAGS.root_path, FLAGS.groundtruth_image_path, FLAGS.sequence)

  osvos_image, _ = davis.io.imread_indexed(osvos_label_path + '00000.png')
  osvos_image = osvos_image[np.newaxis, ...]
  print "osvos_image shape{}, type: {} unique {}".format(osvos_image.shape, type(osvos_image), np.unique(osvos_image))

  maskrcnn_image, _ = davis.io.imread_indexed(maskrcnn_label_path + '00000.png')
  maskrcnn_image = maskrcnn_image[np.newaxis, ...]
  print "maskrcnn_image shape{}, unique {}".format(maskrcnn_image.shape, np.unique(maskrcnn_image))

  groundtruth_label_image, _ = davis.io.imread_indexed(groundtruth_label_path + '00000.png')
  print "groundtruth label shape{}, unique {}".format(groundtruth_label_image.shape, np.unique(groundtruth_label_image))

  groundtruth_image_image = np.concatenate((osvos_image, maskrcnn_image), axis=0)
  print "groundtruth image shape{}, unique {}".format(groundtruth_image_image.shape, np.unique(groundtruth_image_image))
