from __future__ import absolute_import
from __future__ import division

import time
import logging
import os
import sys

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs

logging.basicConfig(level=logging.INFO)


def convolution_layer(filters, kernel=(3, 3), activation='relu', input_shape=None):
  if input_shape is None:
    return tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel, activation=activation)
  else:
    return tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel, activation=activation, input_shape=input_shape)


def concatenated_de_convolution_layer(filters):
  return tf.keras.layers.concatenate(
    [tf.keras.layers.Conv2DTranspose(filters=filters, kernel=(2, 2), strides=(2, 2), padding='same')],
    axis=3)


def max_pooling_layer():
  return tf.keras.layers.MaxPooling2D(pool_size=(2, 2))


# This is a uNet similar neural networks structure
def model_init_fn(FLAGS, inputs):
  # TODO(heguangl): make sure the output dimension is (H x W x 1)
  # TODO(heguangl): reduce network complexity

  input_shape = (FLAGS.height, FLAGS.weight, 2)

  model = tf.keras.models.Sequential()
  model.add(convolution_layer(32, input_shape=input_shape))
  model.add(convolution_layer(32))
  model.add(max_pooling_layer())

  model.add(convolution_layer(64))
  model.add(convolution_layer(64))
  model.add(max_pooling_layer())

  model.add(convolution_layer(128))
  model.add(convolution_layer(128))
  model.add(max_pooling_layer())

  model.add(convolution_layer(256))
  model.add(convolution_layer(256))
  model.add(max_pooling_layer())

  model.add(convolution_layer(512))
  model.add(convolution_layer(512))

  model.add(concatenated_de_convolution_layer(256))
  model.add(convolution_layer(256))
  model.add(convolution_layer(256))

  model.add(concatenated_de_convolution_layer(128))
  model.add(convolution_layer(128))
  model.add(convolution_layer(128))

  model.add(concatenated_de_convolution_layer(64))
  model.add(convolution_layer(64))
  model.add(convolution_layer(64))

  model.add(concatenated_de_convolution_layer(32))
  model.add(convolution_layer(32))
  model.add(convolution_layer(32))

  model.add(convolution_layer(1, kernel=(1, 1), activation='sigmoid'))

  return model(inputs=inputs)


def optimizer_init_fn(FLAGS):
  optimizer = tf.train.AdamOptimizer(1e-5)
  return optimizer
