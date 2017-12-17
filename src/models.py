'''
  File name: models.py
  Author: Haoyuan Zhang
  Date: 12/16/2017
'''

'''
  The file contains models (eg. BaseNet, RPN, fasterRCNN)
'''
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf
import numpy as np
import pdb

from layers import *


'''
  The BaseNet for feature extraction
  - Input view: one of the captured views from 3d object
  - Input input_channel: the number of view channel (eg. 1 (grayscale) or 3(rgb))
  - Input weight_decay_ratio: the weight decay ration, set as 0 if no need to set weight decay
  - Input ind_base: represents the conv block type (0:ConvNet, 1:MobileNet, 2:ResNet)
'''
def BaseNet(view, input_channel, weight_decay_ratio=0.0, ind_base=0, reuse=True, name):
  with tf.variable_scope(name, reuse=reuse) as base:
    # 1st convolution layer
    with tf.variable_scope('conv1', reuse=reuse):
      if ind_base == 0:
        view = ConvBlock(view, input_channel, 32, 5, 1, is_train=True, reuse=reuse, wd=weight_decay_ratio, name='conv1 - type:ConvNet')
      elif ind_base == 1:
        view = MobileBloack(view, input_channel, 32, 5, 5, is_train=True, reuse=reuse, wd=weight_decay_ratio, name='conv1 - type:MobileNet')
      else:
        view = ResBlock(view, input_channel, 32, 3, 3, 1, is_train=True, reuse=reuse, wd=weight_decay_ratio, name='conv1 - type:ResNet')
      
      data_conv1 = maxPool(view, name='max pooling 1')

    # the 2nd convolution layer
    with tf.variable_scope('conv2', reuse=reuse):
      if ind_base == 0:
        data_conv1 = ConvBlock(data_conv1, 32, 64, 5, 1, is_train=True, reuse=reuse, wd=weight_decay_ratio, name='conv2 - type:ConvNet')
      elif ind_base == 1:
        data_conv1 = MobileBloack(data_conv1, 32, 64, 5, 5, is_train=True, reuse=True, wd=weight_decay_ratio, name='conv2 - type:MobileNet')
      else:
        data_conv1 = ResBlock(data_conv1, 32, 64, 3, 3, 1, is_train=True, reuse=reuse, wd=weight_decay_ratio, name='conv2 - type:ResNet')
      
      data_conv2 = maxPool(data_conv1, name='max pooling 2')

    # the 3rd convolution layer
    with tf.variable_scope('conv3', reuse=reuse):
      if ind_base == 0:
        data_conv2 = ConvBlock(data_conv2, 64, 128, 5, 1, is_train=True, reuse=reuse, wd=weight_decay_ratio, name='conv3 - type:ConvNet')
      elif ind_base == 1:
        data_conv2 = MobileBloack(data_conv2, 64, 128, 5, 5, is_train=True, reuse=reuse, wd=weight_decay_ratio, name='conv3 - type:MobileNet')
      else:
        data_conv2 = ResBlock(data_conv2, 64, 128, 3, 3, 1, is_train=True, reuse=reuse, wd=weight_decay_ratio, name='conv3 - type:ResNet')
      
      data_conv3 = maxPool(data_conv2, name='max pooling 3')

    # the 4th convolution layer
    with tf.variable_scope('conv4', reuse=reuse):
      if ind_base == 0:
        data_conv3 = ConvBlock(data_conv3, 64, 128, 5, 1, is_train=True, reuse=reuse, wd=weight_decay_ratio, name='conv4 - type:ConvNet')
      elif ind_base == 1:
        data_conv3 = MobileBloack(data_conv3, 64, 128, 5, 5, is_train=True, reuse=reuse, wd=weight_decay_ratio, name='conv4 - type:MobileNet')
      else:
        data_conv3 = ResBlock(data_conv3, 64, 128, 3, 3, 1, is_train=True, reuse=reuse, wd=weight_decay_ratio, name='conv4 - type:ResNet')


  # obtain all variables of faster rcnn
  var_all = tf.contrib.framework.get_variables(base)

  return var_all, data_conv3