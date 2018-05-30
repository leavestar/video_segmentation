import os
import sys
import numpy as np
import tensorflow as tf
slim = tf.contrib.slim
import davis

def _read_py_function(osvos_file, maskrcnn_file, groundtruth_file):
  osvos_image, _ = davis.io.imread_indexed(osvos_file)
  maskrcnn_image, _ = davis.io.imread_indexed(maskrcnn_file)
  groundtruth_image, _ = davis.io.imread_indexed(groundtruth_file)

  osvos_image = osvos_image[np.newaxis, ...]
  maskrcnn_image = maskrcnn_image[np.newaxis, ...]
  # (2, 480, 854)
  input = np.concatenate((osvos_image, maskrcnn_image), axis=0)
  print "###################{}".format(input.shape)
  return input, groundtruth_image

def load_data(osvos_files, maskrcnn_files, groundtruth_files):
  file_tuple = (osvos_files, maskrcnn_files, groundtruth_files)
  training_dataset = tf.data.Dataset.from_tensor_slices(file_tuple)
  training_dataset = training_dataset.map(
    lambda osvos_file, maskrcnn_file, groundtruth_file : tuple(
      tf.py_func(_read_py_function, [osvos_file, maskrcnn_file, groundtruth_file], [tf.uint8, tf.uint8])
    )
  )
  return training_dataset