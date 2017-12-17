'''
  File name: mainTrainAndTest.py
  Author: Haoyuan Zhang
  Date: 12/16/2017
'''

'''
  The file contains the main training and testing function
'''
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf
import numpy as np
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pdb
from PIL import Image
import time

from layers import *
from utils import *
from models import *
from dataloader import *


################
# Define flags #
################
flags = tf.app.flags
flags.DEFINE_string("logdir", None, "Directory to save logs")
flags.DEFINE_string("sampledir", 'data', "Directory to save cufs samples")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
flags.DEFINE_integer("step", 100, "The step in each epoch iteration [100]")
flags.DEFINE_integer("iteration", 1000, "The max iteration times [1000]")
flags.DEFINE_integer("display_interval", 10, "The step interval to plot training loss and accuracy [10]")
flags.DEFINE_integer("test_interval", 40, "The step interval to test model [40]")
flags.DEFINE_integer("ind_base", 0, "The Conv block ind [0]")
flags.DEFINE_float("weight_decay_ratio", 0.05, "weight decay ration [0.05]")
FLAGS = flags.FLAGS



'''
  fasterRCNN 
  - Input views: NxVxWxHxC (N: batch size, V: number of views)
'''
def fasterRCNN(views, label, bs, reuse, ind_base):
  with tf.variable_scope('fatserRCNN', reuse=reuse) as fasterrcnn:
    # obtain the number of views 
    num_views = views.get_shape().as_list()[1]
    input_channel = views.get_shape().as_list()[-1]

    # transpose views: [NxVxWxHxC] -> [VxNxWxHxC]
    views = tf.transpose(views, perm=[1, 0, 2, 3, 4])

    # multiple views concat
    with tf.variable_scope('multiView_concat', reuse=reuse) as mv:
      view_list = []
      for i in xrange(num_views):
        # set reuse True for i > 0 in order to share weight and bias
        reuse = (i != 0)

        raw_v = tf.gather(views, i)  # obtain raw view with shape [NxWxHxC]

        var_all, feaMap_i = BaseNet(raw_v, input_channel, FLAGS.weight_decay_ratio, ind_base, reuse, name='CNN1')

        # append into feature map list for max pooling
        view_list.append(feaMap_i)

      # max pooling within multiple views 
      feaMap = viewPooling(view_list, 'view pooling')
   
  processPlot(ind_base)
