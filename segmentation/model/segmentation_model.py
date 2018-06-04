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

smoothness = 1.0

def convolution_layer(filters, kernel=(3, 3), activation='relu', input_shape=None):
  if input_shape is None:
    return tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel, activation=activation, padding='same')
  else:
    return tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel, activation=activation, input_shape=input_shape,
                                  padding='same')


def concatenated_de_convolution_layer(filters, input_shape=None):
  if input_shape is None:
    return tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=(2, 2), strides=(2, 2), padding='same')
  else:
    return tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=(2, 2), strides=(2, 2), padding='same',
                                           input_shape=input_shape)


def max_pooling_layer():
  return tf.keras.layers.MaxPooling2D(pool_size=(2, 2))


# This is a uNet similar neural networks structure
def model_init_fn(FLAGS, inputs):
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

  model.add(convolution_layer(1, kernel=(1, 1), activation='relu'))

  return model(inputs=inputs)


def optimizer_init_fn(FLAGS):
  optimizer = tf.train.AdamOptimizer(FLAGS.lr)
  return optimizer


def model_dim_print(FLAGS, channel_dim, inputs):
  input_shape = (FLAGS.height, FLAGS.weight, channel_dim)

  layer_list = []
  if FLAGS.layer32:
    layer_list.append(convolution_layer(32, input_shape=input_shape))
    layer_list.append(convolution_layer(32))
    layer_list.append(max_pooling_layer())

  if FLAGS.layer64:
    layer_list.append(convolution_layer(64))
    layer_list.append(convolution_layer(64))
    layer_list.append(max_pooling_layer())

  if FLAGS.layer128:
    layer_list.append(convolution_layer(128))
    layer_list.append(convolution_layer(128))
    layer_list.append(max_pooling_layer())

  if FLAGS.layer256:
    layer_list.append(convolution_layer(256))
    layer_list.append(convolution_layer(256))
    layer_list.append(max_pooling_layer())

  if FLAGS.layer256:
    layer_list.append(concatenated_de_convolution_layer(256))
    layer_list.append(convolution_layer(256))
    layer_list.append(convolution_layer(256))

  if FLAGS.layer128:
    layer_list.append(concatenated_de_convolution_layer(128))
    layer_list.append(convolution_layer(128))
    layer_list.append(convolution_layer(128))

  if FLAGS.layer64:
    layer_list.append(concatenated_de_convolution_layer(64))
    layer_list.append(convolution_layer(64))
    layer_list.append(convolution_layer(64))

  if FLAGS.layer32:
    layer_list.append(concatenated_de_convolution_layer(32))
    layer_list.append(convolution_layer(32))
    layer_list.append(convolution_layer(32))

  resize_layer = tf.keras.layers.Lambda(lambda image:
                                        tf.image.resize_images(
                                          images=image,
                                          size=[480, 854]
                                        ))

  layer_list.append(resize_layer)
  layer_list.append(convolution_layer(1, kernel=(1, 1), activation='sigmoid'))


  logging.info("Input Layer shape: {}".format(inputs.get_shape()))
  output = []
  output.append(inputs)
  for index in range(len(layer_list)):
    output.append(layer_list[index](output[index]))
    logging.info("Layer {} shape:{}".format(index, output[index + 1].get_shape()))

  model = tf.keras.models.Sequential(layers=layer_list)
  return model(inputs=inputs)

# Not used yet
def dice_coefficient_loss(labels, logits):
  y1 = tf.contrib.layers.flatten(labels)
  y2 = tf.contrib.layers.flatten(logits)
  return - ((2. * tf.reduce_sum(y1 * y2) + smoothness) / (tf.reduce_sum(y1) + tf.reduce_sum(y2) + smoothness))
