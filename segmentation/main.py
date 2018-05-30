import os
import sys
import numpy as np
import tensorflow as tf
slim = tf.contrib.slim
import davis
import matplotlib.pyplot as plt
# import model.segmentation_model.SegmentationModel


# User defined parameters
seq_name = "car-shadow"

train_model = True
# root_path = '/Users/hyuna915/Desktop/2018-CS231N/Final_Project/video_segmentation/davis-2017/data/DAVIS'
# osvos_label_path = 'Results/Segmentations/480p/OSVOS/'
# maskrcnn_label_path = 'MaskRCNN/480p/'
# groundtruth_label_path= 'Annotations/480p/'
# groundtruth_image_path = 'JPEGImages/480p/'

tf.app.flags.DEFINE_string("sequence", "elephant", "which sequence")
tf.app.flags.DEFINE_string("root_path", "/Users/hyuna915/Desktop/2018-CS231N/Final_Project/video_segmentation/davis-2017/data/DAVIS", "Available modes: train / show_examples / official_eval")
tf.app.flags.DEFINE_string("maskrcnn_label_path", "MaskRCNN/480p", "")
tf.app.flags.DEFINE_string("osvos_label_path", "Results/Segmentations/480p/OSVOS", "")
tf.app.flags.DEFINE_string("groundtruth_label_path", "Annotations/480p", "groundtruth_label_path")
tf.app.flags.DEFINE_string("groundtruth_image_path", "JPEGImages/480p", "groundtruth_image_path")
tf.app.flags.DEFINE_integer("height", 480, "height")
tf.app.flags.DEFINE_integer("weight", 854, "weight")

tf.app.flags.DEFINE_integer("filter", 64, "weight")
tf.app.flags.DEFINE_integer("kernel", 3, "weight")
tf.app.flags.DEFINE_integer("pad", 1, "weight")
tf.app.flags.DEFINE_integer("pool", 2, "weight")


max_training_iters = 500
FLAGS = tf.app.flags.FLAGS


def main(unused_argv):
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

  groundtruth_label_image, _ = davis.io.imread_indexed(groundtruth_label_path+'00000.png')
  print "groundtruth label shape{}, unique {}".format(groundtruth_label_image.shape, np.unique(groundtruth_label_image))

  groundtruth_image_image = np.concatenate((osvos_image, maskrcnn_image), axis=0)
  print "groundtruth image shape{}, unique {}".format(groundtruth_image_image.shape, np.unique(groundtruth_image_image))





if __name__ == "__main__":
    tf.app.run()
