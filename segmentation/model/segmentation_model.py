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

def model_init_fn(FLAGS, inputs, is_training=False):
  input_shape = (2, FLAGS.height, FLAGS.weight)
  kernel = FLAGS.kernel
  filter_size = 64
  pad = 1
  pool_size = 2

  model = tf.keras.models.Sequential()
  model.add(tf.keras.layers.Layer(input_shape=input_shape))
  model.add(tf.keras.layers.ZeroPadding2D(padding=(pad, pad)))
  model.add(tf.keras.layers.Conv2D(filter_size, kernel, kernel, border_mode='valid'))
  model.add(tf.keras.layers.BatchNormalization())
  model.add(tf.keras.layers.Activation('relu'))
  model.add(tf.keras.layers.MaxPooling2D(pool_size=(pool_size, pool_size)))

  model.add(tf.keras.layers.UpSampling2D(size=(pool_size, pool_size)))
  model.add(tf.keras.layers.ZeroPadding2D(padding=(pad, pad)))
  model.add(tf.keras.layers.Conv2D(filter_size, kernel, kernel, border_mode='valid'))
  model.add(tf.keras.layers.BatchNormalization())
  model.add(tf.keras.layers.Conv2D(1, 1, 1, border_mode='valid', ))
  model.outputHeight = model.output_shape[-2]
  model.outputWidth = model.output_shape[-1]
  model.add(tf.keras.layers.Reshape((1, model.output_shape[-2] * model.output_shape[-1]),
                                    input_shape=(1, model.output_shape[-2], model.output_shape[-1])))
  model.add(tf.keras.layers.Permute((2, 1)))
  model.add(tf.keras.layers.Activation('softmax'))
  model.compile(loss="categorical_crossentropy", optimizer=tf.keras.optimizers.Adam, metrics=['accuracy'])
  return model(inputs)

def optimizer_init_fn(learning_rate):
  optimizer = tf.train.AdamOptimizer(learning_rate)
  return optimizer



