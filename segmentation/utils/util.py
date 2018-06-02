import os
import re
import sys
import numpy as np
import logging
import tensorflow as tf
import davis

logging.basicConfig(level=logging.DEBUG)

def load_image_files(FLAGS, seqs):
  osvos_label_paths = []
  maskrcnn_label_paths = []
  groundtruth_label_paths = []
  groundtruth_image_paths = []

  for sequence in seqs:
    osvos_label_path = "{}/{}/{}/".format(FLAGS.read_path, FLAGS.osvos_label_path, sequence)
    maskrcnn_label_path = "{}/{}/{}/".format(FLAGS.read_path, FLAGS.maskrcnn_label_path, sequence)
    groundtruth_label_path = "{}/{}/{}/".format(FLAGS.read_path, FLAGS.groundtruth_label_path, sequence)
    groundtruth_image_path = "{}/{}/{}/".format(FLAGS.read_path, FLAGS.groundtruth_image_path, sequence)

    for file in os.listdir(groundtruth_label_path):
      if re.match(r'[0-9]+.*\.png', file):
        osvos_label_paths.append(osvos_label_path + file)
        maskrcnn_label_paths.append(maskrcnn_label_path + file)
        groundtruth_label_paths.append(groundtruth_label_path + file)
        logging.debug(groundtruth_label_path + file)
        logging.debug(osvos_label_path + file)
        logging.debug(maskrcnn_label_path + file)

  assert len(osvos_label_paths) == len(maskrcnn_label_paths)
  assert len(maskrcnn_label_paths) == len(groundtruth_label_paths)
  return osvos_label_paths, maskrcnn_label_paths, groundtruth_label_paths


def check_image_dimension(FLAGS):
  osvos_label_path = "{}/{}/{}/".format(FLAGS.read_path, FLAGS.osvos_label_path, FLAGS.sequence)
  maskrcnn_label_path = "{}/{}/{}/".format(FLAGS.read_path, FLAGS.maskrcnn_label_path, FLAGS.sequence)
  groundtruth_label_path = "{}/{}/{}/".format(FLAGS.read_path, FLAGS.groundtruth_label_path, FLAGS.sequence)
  groundtruth_image_path = "{}/{}/{}/".format(FLAGS.read_path, FLAGS.groundtruth_image_path, FLAGS.sequence)

  osvos_image, _ = davis.io.imread_indexed(osvos_label_path + '00000.png')
  osvos_image = osvos_image[np.newaxis, ...]
  logging.debug("osvos_image shape{}, type: {} unique {}".format(osvos_image.shape, type(osvos_image), np.unique(osvos_image)))

  maskrcnn_image, _ = davis.io.imread_indexed(maskrcnn_label_path + '00000.png')
  maskrcnn_image = maskrcnn_image[np.newaxis, ...]
  logging.debug("maskrcnn_image shape{}, unique {}".format(maskrcnn_image.shape, np.unique(maskrcnn_image)))

  groundtruth_label_image, _ = davis.io.imread_indexed(groundtruth_label_path + '00000.png')
  logging.debug("groundtruth label shape{}, unique {}".format(groundtruth_label_image.shape, np.unique(groundtruth_label_image)))

  groundtruth_image_image = np.concatenate((osvos_image, maskrcnn_image), axis=0)
  logging.debug("groundtruth image shape{}, unique {}".format(groundtruth_image_image.shape, np.unique(groundtruth_image_image)))
