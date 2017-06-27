#!/usr/bin/env python

import argparse
import os
from random import shuffle
import shutil
import subprocess
import sys

HOMEDIR = os.path.expanduser("~")
CURDIR = os.path.dirname(os.path.realpath(__file__))

"""Dataset structure
- cones
  - train
    - images
    - labels
  - val
    - images
    - labels
"""
data_dir = "{}/data/cones".format(HOMEDIR)

# To create LMDB, there must be a txt file, every row of that file 
# indicates a image path and the corespoding label path

img_ext = '.png'
label_ext = '.txt'


for split in ['train', 'val']:
    im_dir = os.path.join(data_dir, split, 'images')
    label_dir = os.path.join(data_dir, split, 'labels')
    final_list = []
    files = []
    for (dirpath, dirnames, filenames) in os.walk(im_dir):
        files.extend(filenames)
        break
    for im_name in files:
      im_base_name = os.path.splitext(im_name)[0]
      label_name = im_base_name + label_ext
      im_path = os.path.join('{}/images'.format(split), im_name)
      label_path = os.path.join('{}/labels'.format(split), label_name)
      final_list.append('{} {}\n'.format(im_path, label_path))

    if split == 'train':
      shuffle(final_list)


    list_file = "{}/{}.txt".format(CURDIR, split)
    with open(list_file, 'w') as lf:
      for line in final_list:
        lf.write(line)
