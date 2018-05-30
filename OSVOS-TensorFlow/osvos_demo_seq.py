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
from PIL import Image
import numpy as np
import tensorflow as tf
import yaml
slim = tf.contrib.slim
import matplotlib.pyplot as plt
# Import OSVOS files
root_folder = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(root_folder))
import osvos
from dataset import Dataset

# import davis tool
from davis import *

os.chdir(root_folder)


# User defined parameters
gpu_id = 0
train_model = True

# Train parameters
parent_path = os.path.join('models', 'OSVOS_parent', 'OSVOS_parent.ckpt-50000')
max_training_iters = 500


CUSTOM_ANNOTATION_DIR = os.path.join('..', 'davis-2017', 'data','DAVIS', 'cusom_Annotations')
def demo(seq_name):
    # first read in ground truth segmentation and determine how many object are there
    annatation_0 = os.path.join('..', 'davis-2017', 'data','DAVIS', 'Annotations', '480p', seq_name, '00000.png')
    image_0 = os.path.join('..', 'davis-2017', 'data', 'DAVIS', 'JPEGImages', '480p', seq_name, '00000.jpg')
    an, _ = io.imread_indexed(annatation_0)
    N_OBJECT = len(np.unique(an)) - 1

    for n_object_ in range(1, N_OBJECT+1):
        n_object = str(n_object_)
        result_path = os.path.join('..', 'davis-2017', 'data', 'DAVIS', 'Results', 'Segmentations', '480p', 'OSVOS2', seq_name, n_object)
        logs_path = os.path.join('models2', seq_name, n_object)

        # Define Dataset
        test_frames = sorted(os.listdir(os.path.join('..', 'davis-2017', 'data','DAVIS', 'JPEGImages', '480p', seq_name)))
        test_imgs = [os.path.join('..', 'davis-2017', 'data','DAVIS', 'JPEGImages', '480p', seq_name, frame) for frame in test_frames]
        if train_model:
            # we need to first create a new annotation that has only one object
            base_image = np.zeros_like(an).astype('uint8')
            base_image[an == n_object_] = n_object_
            custom_anno = os.path.join(CUSTOM_ANNOTATION_DIR, seq_name, n_object)
            if not os.path.exists(custom_anno):
                os.makedirs(custom_anno)
            io.imwrite_indexed(os.path.join(custom_anno, "00000.png"), base_image)

            train_imgs = [image_0 + ' ' + os.path.join(custom_anno, "00000.png")]
            dataset = Dataset(train_imgs, test_imgs, './', data_aug=True)
        else:
            dataset = Dataset(None, test_imgs, './')

        # Train the network
        if train_model:
            # More training parameters
            learning_rate = 1e-8
            save_step = max_training_iters
            side_supervision = 3
            display_step = 10
            with tf.Graph().as_default():
                with tf.device('/gpu:' + str(gpu_id)):
                # with tf.device('/cpu:0'):
                #     import pdb; pdb.set_trace()
                    global_step = tf.Variable(0, name='global_step', trainable=False)
                    osvos.train_finetune(dataset, parent_path, side_supervision, learning_rate, logs_path, max_training_iters,
                                         save_step, display_step, global_step, iter_mean_grad=1, ckpt_name=seq_name)

        # import pdb; pdb.set_trace()
        # Test the network
        with tf.Graph().as_default():
            with tf.device('/gpu:' + str(gpu_id)):
                checkpoint_path = os.path.join('models2', seq_name, n_object, seq_name+'.ckpt-'+str(max_training_iters))
                osvos.test(dataset, checkpoint_path, result_path)

# Show results
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
path = os.path.join('..', 'davis-2017', 'data', 'db_info_test.yaml')
stream = file(path, 'r')
dict = yaml.load(stream)
seq = dict['sequences']
for item in seq:
    print item['name']
    demo(item['name'])

