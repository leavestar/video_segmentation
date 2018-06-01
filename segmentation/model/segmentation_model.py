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
    return tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel, activation=activation, padding='same')
  else:
    return tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel, activation=activation, input_shape=input_shape, padding='same')

#
# def concatenated_de_convolution_layer(filters):
#   return tf.keras.layers.concatenate(
#     [tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=(2, 2), strides=(2, 2), padding='same')],
#     axis=3)

def concatenated_de_convolution_layer(filters, input_shape=None):
  if input_shape is None:
    return tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=(2, 2), strides=(2, 2), padding='same')
  else:
    return tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=(2, 2), strides=(2, 2), padding='same', input_shape=input_shape)


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

  model.add(convolution_layer(1, kernel=(1, 1), activation='sigmoid'))

  return model(inputs=inputs)


def optimizer_init_fn(FLAGS):
  optimizer = tf.train.AdamOptimizer(1e-5)
  return optimizer


def model_dim_print(FLAGS, inputs):

  input_shape = (FLAGS.height, FLAGS.weight, 2)

  layer_list=[]
  layer_list.append(convolution_layer(32, input_shape=input_shape))
  layer_list.append(convolution_layer(32))
  layer_list.append(max_pooling_layer())

  layer_list.append(convolution_layer(64))
  layer_list.append(convolution_layer(64))
  layer_list.append(max_pooling_layer())

  layer_list.append(concatenated_de_convolution_layer(64))
  layer_list.append(convolution_layer(64))
  layer_list.append(convolution_layer(64))

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

  output = []
  output.append(inputs)
  for index in range(len(layer_list)):
    output.append(layer_list[index](output[index]))
    logging.debug("Layer {} shape:{}".format(index, output[index+1].get_shape()))

  model = tf.keras.models.Sequential(layers=layer_list)
  return model(inputs=inputs)
