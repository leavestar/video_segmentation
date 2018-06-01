import os
import sys

import numpy as np
import tensorflow as tf
slim = tf.contrib.slim
import davis
import logging

logging.basicConfig(level=logging.INFO)

def _read_py_function(osvos_file, maskrcnn_file, groundtruth_file):
  osvos_image, _ = davis.io.imread_indexed(osvos_file)
  maskrcnn_image, _ = davis.io.imread_indexed(maskrcnn_file)
  groundtruth_image, _ = davis.io.imread_indexed(groundtruth_file)

  osvos_image = osvos_image[..., np.newaxis ]
  maskrcnn_image = maskrcnn_image[..., np.newaxis]
  groundtruth_image = groundtruth_image[..., np.newaxis]
  # (480, 854, 2)
  input = np.concatenate((osvos_image.astype(np.float32), maskrcnn_image.astype(np.float32)), axis=2)
  groundtruth_image = groundtruth_image.astype(np.float32)
  logging.debug("################### input shape {} type {} dtype {}".format(input.shape, type(input), input.dtype))
  logging.debug("################### groundtruth_label shape {} type {} dtype {}".format(groundtruth_image.shape, type(groundtruth_image), groundtruth_image.dtype))
  return input, groundtruth_image

def load_data(osvos_files, maskrcnn_files, groundtruth_files):
  file_tuple = (osvos_files, maskrcnn_files, groundtruth_files)
  training_dataset = tf.data.Dataset.from_tensor_slices(file_tuple)
  training_dataset = training_dataset.map(
    lambda osvos_file, maskrcnn_file, groundtruth_file : tuple(
      tf.py_func(_read_py_function, [osvos_file, maskrcnn_file, groundtruth_file], [tf.float32, tf.float32])
    )
  )
  logging.debug("Dataset type{},  shape {}, classes {}".format(
    training_dataset.output_types,
    training_dataset.output_shapes,
    training_dataset.output_classes))
  return training_dataset