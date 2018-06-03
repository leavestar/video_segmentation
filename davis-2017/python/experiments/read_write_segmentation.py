#!/usr/bin/env python

# ----------------------------------------------------------------------------
# The 2017 DAVIS Challenge on Video Object Segmentation
#-----------------------------------------------------------------------------
# Copyright (c) 2017 Federico Perazzi
# Licensed under the BSD License [see LICENSE for details]
# Written by Federico Perazzi
# ----------------------------------------------------------------------------

"""
Read and write segmentation in indexed format.

EXAMPLE:
    python experiments/read_write_segmentation.py

"""
import os
import cv2
from   davis import cfg,io,DAVISLoader
from collections import Counter
import numpy as np
# Load dataset
# db = DAVISLoader(year=cfg.YEAR, phase=cfg.PHASE)

# Save an annotation in PNG indexed format to a temporary file
# io.imwrite_indexed('/tmp/anno_indexed.png', db[0].annotations[0])
max_, max_seq, cnt = 0, "", Counter()
for seq_name in cfg.SEQUENCES.keys():
  annatation_0 = os.path.join('../../', 'data', 'DAVIS', 'Annotations', '480p', seq_name, '00000.png')
  an, _ = io.imread_indexed(annatation_0)
  cnt[seq_name] = np.max(an)
  if max_<np.max(an):
    max_ = np.max(an)
    max_seq = seq_name
print max_, max_seq
for _ in cnt.most_common():
  print _

# small script to find largest # of object in the dataset



# # Read an image in a temporary file
# seq_name = "drone"
# annatation_0 = os.path.join('../../', 'data', 'DAVIS', 'Annotations', '480p', seq_name, '00000.png')
# an,_ = io.imread_indexed(annatation_0)
# # an,_ = io.imread_indexed('/tmp/anno_indexed.png')
#
# # test on changing and write back
# base_image = np.zeros_like(an).astype('uint8')
# base_image[an == 3] = 3
# io.imwrite_indexed('/tmp/anno_indexed.png', base_image)
#
# test_result,_ = io.imread_indexed('/tmp/anno_indexed.png')
#
# cv2.imshow("Segmentation",cfg.palette[an][...,[2,1,0]])
# cv2.waitKey()
