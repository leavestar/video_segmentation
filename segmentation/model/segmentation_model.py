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

class VideoSegmentation(tf.keras.Model):
  def __int__(self, FLAGS, channel1, channel2, channel3):


