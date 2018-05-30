"""
Sergi Caelles (scaelles@vision.ee.ethz.ch)

This file is part of the OSVOS paper presented in:
    Sergi Caelles, Kevis-Kokitsi Maninis, Jordi Pont-Tuset, Laura Leal-Taixe, Daniel Cremers, Luc Van Gool
    One-Shot Video Object Segmentation
    CVPR 2017
Please consider citing the paper if you use this code.
"""
import os
import sys
import numpy as np
import tensorflow as tf
slim = tf.contrib.slim
import skimage.color
import skimage.io
import skimage.transform
import matplotlib.pyplot as plt


# User defined parameters
seq_name = "elephant"

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
tf.app.flags.DEFINE_string("groundtruth_label_path", "Annotations/480p", "")
tf.app.flags.DEFINE_string("groundtruth_image_path", "JPEGImages/480p", "")

max_training_iters = 500
FLAGS = tf.app.flags.FLAGS
osvos_label_path = "{}/{}/{}/".format(FLAGS.root_path, FLAGS.osvos_label_path, FLAGS.sequence)
print osvos_label_path
osvos_image = skimage.io.imread(osvos_label_path+'00000.png')
print "osvos_image shape{}, unique {}".format(osvos_image.shape, np.unique(osvos_image))

maskrcnn_label_path = "{}/{}/{}/".format(FLAGS.root_path, FLAGS.maskrcnn_label_path, FLAGS.sequence)
maskrcnn_image = skimage.io.imread(maskrcnn_label_path+'00000.png')
print "maskrcnn_image shape{}, unique {}".format(maskrcnn_image.shape, np.unique(maskrcnn_image))

groundtruth_label_path = "{}/{}/{}/".format(FLAGS.root_path, FLAGS.groundtruth_label_path, FLAGS.sequence)
groundtruth_label_image = skimage.io.imread(groundtruth_label_path+'00000.png')
print "groundtruth label shape{}, unique {}".format(groundtruth_label_image.shape, np.unique(groundtruth_label_image))

groundtruth_image_path = "{}/{}/{}/".format(FLAGS.root_path, FLAGS.groundtruth_image_path, FLAGS.sequence)
groundtruth_image_image = skimage.io.imread(groundtruth_image_path+'00000.jpg')
print "groundtruth label shape{}, unique {}".format(groundtruth_image_image.shape, np.unique(groundtruth_image_image))


#
# # Train the network
# if train_model:
#     # More training parameters
#     learning_rate = 1e-8
#     save_step = max_training_iters
#     side_supervision = 3
#     display_step = 10
#     with tf.Graph().as_default():
#         with tf.device('/gpu:' + str(gpu_id)):
#             global_step = tf.Variable(0, name='global_step', trainable=False)
#             osvos.train_finetune(dataset, parent_path, side_supervision, learning_rate, logs_path, max_training_iters,
#                                  save_step, display_step, global_step, iter_mean_grad=1, ckpt_name=seq_name)
#
# # Test the network
# with tf.Graph().as_default():
#     with tf.device('/gpu:' + str(gpu_id)):
#         checkpoint_path = os.path.join('models', seq_name, seq_name+'.ckpt-'+str(max_training_iters))
#         osvos.test(dataset, checkpoint_path, result_path)
#
# # Show results
# overlay_color = [255, 0, 0]
# transparency = 0.6
# plt.ion()
# for img_p in test_frames:
#     frame_num = img_p.split('.')[0]
#     img = np.array(Image.open(os.path.join('DAVIS', 'JPEGImages', '480p', seq_name, img_p)))
#     mask = np.array(Image.open(os.path.join(result_path, frame_num+'.png')))
#     mask = mask/np.max(mask)
#     im_over = np.ndarray(img.shape)
#     im_over[:, :, 0] = (1 - mask) * img[:, :, 0] + mask * (overlay_color[0]*transparency + (1-transparency)*img[:, :, 0])
#     im_over[:, :, 1] = (1 - mask) * img[:, :, 1] + mask * (overlay_color[1]*transparency + (1-transparency)*img[:, :, 1])
#     im_over[:, :, 2] = (1 - mask) * img[:, :, 2] + mask * (overlay_color[2]*transparency + (1-transparency)*img[:, :, 2])
#     plt.imshow(im_over.astype(np.uint8))
#     plt.axis('off')
#     plt.show()
#     plt.pause(0.01)
#     plt.clf()
