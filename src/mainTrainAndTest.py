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


    # Cls branch with variable space: cls_branch
    with tf.variable_scope('cls_branch', reuse=reuse) as cls_branch:
      # get cls feature map
      with tf.variable_scope('clsMap', reuse=reuse):
        Weight = tf.Variable(tf.truncated_normal([1, 1, 256, 1], stddev=0.1), name='W_cls')
        Weight = varWeightDecay(Weight, weight_decay_ratio)  # weight decay
        bias = tf.Variable(tf.constant(0.1, shape=[1]), name='b_cls')
        cls_map = conv2d(inter_map, Weight) + bias
        cls_map = tf.reshape(cls_map, [-1, 6, 6])

      # compute cls loss (sigmoid cross entropy)
      with tf.variable_scope('clsLoss', reuse=reuse):
        # obtain valid places (pos and neg) in mask
        cond_cls = tf.not_equal(mask_batch, tf.constant(2, dtype=tf.float32))
        # compute the sigmoid cross entropy loss: choose loss where cond is 1 while select 0
        cross_entropy_classify = tf.reduce_sum(
          tf.where(cond_cls, tf.nn.sigmoid_cross_entropy_with_logits(labels=mask_batch, logits=cls_map), tf.zeros_like(mask_batch)))
        # count the pos and neg numbers
        effect_area_cls = tf.reduce_sum(tf.where(cond_cls, tf.ones_like(mask_batch), tf.zeros_like(mask_batch)))
        cross_entropy_classify /= effect_area_cls

      # compute cls accuracy
      with tf.variable_scope('clsAcc', reuse=reuse):
        correct = tf.reduce_sum(tf.where(cond_cls, tf.cast(abs(mask_batch - tf.nn.sigmoid(cls_map)) < 0.5, tf.float32), tf.zeros_like(mask_batch)))
        effect_area = tf.reduce_sum(tf.where(cond_cls, tf.ones_like(mask_batch), tf.zeros_like(mask_batch)))
        cls_acc = correct / effect_area


    # fully connected layer: convert from 4x4x256 to 256
    with tf.variable_scope('FC1', reuse=reuse) as fc1:
      fc_map = tf.reshape(spatial_map, [bs, 4*4*256])
      fc_map_flat = FcBlock(fc_map, 4*4*256, 256, is_train=True, reuse=reuse, wd=weight_decay_ratio)

    # fully connected layer: convert from 256 to 10
    with tf.variable_scope('FC2', reuse=reuse) as fc2:
      # fc_map_flat = tf.reshape(fc_map, [bs, 256])
      W = tf.get_variable('weights', [256, 10], initializer=tf.truncated_normal_initializer(stddev=0.1))
      W = varWeightDecay(W, weight_decay_ratio)  # weight decay
      pred = tf.matmul(fc_map_flat, W)

    # softmax layer
    with tf.variable_scope('softmax', reuse=reuse) as sm:
      sf_loss = tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=tf.one_hot(label, 10))  # one_hot make the label same size as pred
      sf_loss = tf.reduce_mean(sf_loss)
      sf_acc = tf.reduce_mean(tf.to_float(tf.equal(tf.argmax(pred, axis=1), label)))

  # joint loss    
  comb_loss = 10 * smooth_L1_loss_reg + cross_entropy_classify + sf_loss
  # obtain all variables of faster rcnn
  var_fastrcnn = tf.contrib.framework.get_variables(fasterrcnn)

  return comb_loss, sf_loss, sf_acc, cross_entropy_classify, cls_acc, smooth_L1_loss_reg, reg_acc, var_fastrcnn


'''
  The train and test model
'''
def trainAndTest(dict_train, dict_test, ind_base):
  print 'The current base net is [' + net_label[ind_base] + ']'

  best_test_rcnn_loss, best_test_rcnn_acc = 100, 0
  best_test_cls_loss, best_test_cls_acc = 100, 0
  best_test_reg_loss, best_test_reg_acc = 100, 0


  image = tf.placeholder(tf.float32, [batch_size, 48, 48, 3])  # x_image represents the input image
  mask_gt = tf.placeholder(tf.float32, [batch_size, 6, 6])  # mask for classification
  reg_gt = tf.placeholder(tf.float32, [batch_size, 6, 6, 3])  # the ground truth for proposal regression
  label_gt = tf.placeholder(tf.int64, [batch_size])  # the ground truth label

  # faster rcnn model
  comb_loss, sf_loss, sf_acc, cls_loss, cls_acc, reg_loss, reg_acc, var_frcnn = \
                                      fasterRCNN(image, mask_gt, reg_gt, label_gt, batch_size, None, ind_base)

  # define learning rate decay parameters
  global_step = tf.Variable(0, trainable=False)
  starter_learning_rate = 0.001
  learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, step, 1 - 10 ** (-iteration), staircase=True)

  # train_step = tf.train.AdamOptimizer(1e-4).minimize(loss, var_list=var_all)
  train_step = tf.train.AdamOptimizer(learning_rate).minimize(comb_loss, var_list=var_frcnn, global_step=global_step)

  list_loss_rcnn, list_acc_rcnn = np.zeros([iteration, 1]), np.zeros([iteration, 1])
  list_loss_cls, list_acc_cls = np.zeros([iteration, 1]), np.zeros([iteration, 1])
  list_loss_reg, list_acc_reg = np.zeros([iteration, 1]), np.zeros([iteration, 1])

  sess = tf.InteractiveSession()
  sess.run(tf.global_variables_initializer())

  epoch, time_count = 0, 0
  while (epoch < iteration):
    print '\n********************************* The {}th epoch training is processing *********************************'.format(epoch + 1)
    start_time = time.time()
    # data shuffle 
    arr = np.arange(dict_train['img'].shape[0])
    np.random.shuffle(arr)

    # batch size data
    data_epoch = dict_train['img'][arr[:], :, :, :]
    mask_epoch = dict_train['cls_gt'][arr[:], :, :]
    reg_mask_epoch = dict_train['reg_gt'][arr[:], :, :, :]
    label_epoch = dict_train['label'][arr[:]]

    # train for each epoch
    cls_loss_sum, reg_loss_sum, rcnn_loss_sum = 0, 0, 0
    cls_acc_sum, reg_acc_sum, rcnn_acc_sum = 0, 0, 0

    step_cur = 0
    while (step_cur < step):
      # obtain current batch data for training
      data_batch = data_epoch[batch_size*step_cur : batch_size*(step_cur+1), :, :, :]
      mask_batch = mask_epoch[batch_size*step_cur : batch_size*(step_cur+1), :, :]
      reg_mask_batch = reg_mask_epoch[batch_size*step_cur : batch_size*(step_cur+1), :, :, :]
      label_batch = label_epoch[batch_size*step_cur : batch_size*(step_cur+1)]

      # evaluation
      train_loss_rcnn = sf_loss.eval(feed_dict={image:data_batch, mask_gt:mask_batch, reg_gt:reg_mask_batch, label_gt:label_batch})
      train_acc_rcnn = sf_acc.eval(feed_dict={image:data_batch, mask_gt:mask_batch, reg_gt:reg_mask_batch, label_gt:label_batch})

      train_loss_cls = cls_loss.eval(feed_dict={image:data_batch, mask_gt:mask_batch, reg_gt:reg_mask_batch, label_gt:label_batch})
      train_acc_cls = cls_acc.eval(feed_dict={image:data_batch, mask_gt:mask_batch, reg_gt:reg_mask_batch, label_gt:label_batch})

      train_loss_reg = reg_loss.eval(feed_dict={image:data_batch, mask_gt:mask_batch, reg_gt:reg_mask_batch, label_gt:label_batch})
      train_acc_reg = reg_acc.eval(feed_dict={image:data_batch, mask_gt:mask_batch, reg_gt:reg_mask_batch, label_gt:label_batch})

      rcnn_loss_sum += train_loss_rcnn
      cls_loss_sum += train_loss_cls
      reg_loss_sum += train_loss_reg

      rcnn_acc_sum += train_acc_rcnn
      cls_acc_sum += train_acc_cls
      reg_acc_sum += train_acc_reg

      if step_cur % display_interval == 0:
        print('Train Epoch:{} [{}/{}]  Obj Loss: {:.8f}, Accuracy: {:.2f}% || Cls Loss: {:.8f}, Accuracy: {:.2f}% || Reg Loss: {:.8f}, Accuracy: {:.4f}%'.format(
          epoch + 1, step_cur * batch_size , step * batch_size, train_loss_rcnn, 100 * train_acc_rcnn, train_loss_cls, 100. * train_acc_cls, train_loss_reg, 100. * train_acc_reg))

      # train the model
      train_step.run(feed_dict={image:data_batch, mask_gt:mask_batch, reg_gt:reg_mask_batch, label_gt:label_batch})
      step_cur += 1

    elapsed_time = time.time() - start_time
    time_count += elapsed_time
    # end of current epoch iteration 
    
    rcnn_loss_sum /= step
    cls_loss_sum /= step
    reg_loss_sum /= step

    rcnn_acc_sum /= step
    cls_acc_sum /= step
    reg_acc_sum /= step
    # print the result for each epoch
    print '\n********************************* The {}th epoch training has completed using {} s *********************************'.format(epoch + 1, elapsed_time)
    print 'The Avg Obj Loss is {:.8f}, Avg Acc is {:.2f}% || The Avg Cls Loss is {:.8f}, Avg Acc is {:.2f}% || The Avg Reg Loss is {:.8f}, Avg Acc is {:.2f}%. \n'.format(
      rcnn_loss_sum, 100 * rcnn_acc_sum, cls_loss_sum, 100 * cls_acc_sum, reg_loss_sum, 100 * reg_acc_sum)

    # store results
    list_loss_rcnn[epoch, 0], list_acc_rcnn[epoch, 0] = rcnn_loss_sum, rcnn_acc_sum
    list_loss_cls[epoch, 0], list_acc_cls[epoch, 0] = cls_loss_sum, cls_acc_sum
    list_loss_reg[epoch, 0], list_acc_reg[epoch, 0] = reg_loss_sum, reg_acc_sum


    '''
      test model
    '''
    test_rcnn_loss, test_rcnn_acc = 0, 0
    test_cls_loss, test_cls_acc, test_reg_loss, test_reg_acc = 0, 0, 0, 0
    if (epoch + 1) % test_interval == 0:
      print '==========================================================================================================='
      print '--------------------------- [TEST] the trained model after {} epochs --------------------------------------'.format(epoch + 1)
      test_data_epoch = dict_test['img']
      test_mask_epoch = dict_test['cls_gt']
      test_reg_mask_epoch = dict_test['reg_gt']
      test_label_epoch = dict_test['label']

      step_cur = 0
      while (step_cur < step):
        # obtain current batch data for training
        test_data_batch = test_data_epoch[batch_size*step_cur : batch_size*(step_cur+1), :, :, :]
        test_mask_batch = test_mask_epoch[batch_size*step_cur : batch_size*(step_cur+1), :, :]
        test_reg_mask_batch = test_reg_mask_epoch[batch_size*step_cur : batch_size*(step_cur+1), :, :, :]
        test_label_batch = test_label_epoch[batch_size*step_cur : batch_size*(step_cur+1)]

        # evaluation
        test_rcnn_loss += sf_loss.eval(feed_dict={image:test_data_batch, mask_gt:test_mask_batch, reg_gt:test_reg_mask_batch, label_gt:test_label_batch})
        test_rcnn_acc += sf_acc.eval(feed_dict={image:test_data_batch, mask_gt:test_mask_batch, reg_gt:test_reg_mask_batch, label_gt:test_label_batch})
        
        test_cls_loss += cls_loss.eval(feed_dict={image:test_data_batch, mask_gt:test_mask_batch, reg_gt:test_reg_mask_batch, label_gt:test_label_batch})
        test_cls_acc += cls_acc.eval(feed_dict={image:test_data_batch, mask_gt:test_mask_batch, reg_gt:test_reg_mask_batch, label_gt:test_label_batch})

        test_reg_loss += reg_loss.eval(feed_dict={image:test_data_batch, mask_gt:test_mask_batch, reg_gt:test_reg_mask_batch, label_gt:test_label_batch})
        test_reg_acc += reg_acc.eval(feed_dict={image:test_data_batch, mask_gt:test_mask_batch, reg_gt:test_reg_mask_batch, label_gt:test_label_batch})

        step_cur += 1

      # end test
      test_rcnn_loss /= step
      test_rcnn_acc /= step
      test_cls_loss /= step
      test_cls_acc /= step
      test_reg_loss /= step
      test_reg_acc /= step

      # store results
      if test_rcnn_acc > best_test_rcnn_acc:
        best_test_rcnn_acc = test_rcnn_acc
        best_test_rcnn_loss = test_rcnn_loss
        best_test_cls_acc = test_cls_acc
        best_test_cls_loss = test_cls_loss
        best_test_reg_acc = test_reg_acc
        best_test_reg_loss = test_reg_loss


      print 'The Test Obj Loss is {:.8f}, Acc is {:.2f}%, The [Best Acc] so far is {:.2f}%. \nThe Test Cls Loss is {:.8f}, Acc is {:.2f}% || The Test Reg Loss is {:.8f}, Acc is {:.2f}%.'.format(
        test_rcnn_loss, 100 * test_rcnn_acc, 100 * best_test_rcnn_acc, test_cls_loss, 100 * test_cls_acc, test_reg_loss, 100 * test_reg_acc)
      print '==========================================================================================================='

      np.save('res/FasterRCNN{}_test_rcnn.npy'.format(ind_base), [best_test_rcnn_loss, best_test_rcnn_acc])
      np.save('res/FasterRCNN{}_test_cls.npy'.format(ind_base), [best_test_cls_loss, best_test_cls_acc])
      np.save('res/FasterRCNN{}_test_reg.npy'.format(ind_base), [best_test_reg_loss, best_test_reg_acc])

      np.save('res/FasterRCNN{}_train_rcnn_loss.npy'.format(ind_base), list_loss_rcnn[:epoch + 1, :])
      np.save('res/FasterRCNN{}_train_rcnn_acc.npy'.format(ind_base), list_acc_rcnn[:epoch + 1, :])
      np.save('res/FasterRCNN{}_train_cls_loss.npy'.format(ind_base), list_loss_cls[:epoch + 1, :])
      np.save('res/FasterRCNN{}_train_cls_acc.npy'.format(ind_base), list_acc_cls[:epoch + 1, :])
      np.save('res/FasterRCNN{}_train_reg_loss.npy'.format(ind_base), list_loss_reg[:epoch + 1, :])
      np.save('res/FasterRCNN{}_train_reg_acc.npy'.format(ind_base), list_acc_reg[:epoch + 1, :])

    # update epoch
    epoch += 1

    # store results
    np.save('res/FasterRCNN{}_test_rcnn.npy'.format(ind_base), [best_test_rcnn_loss, best_test_rcnn_acc])
    np.save('res/FasterRCNN{}_test_cls.npy'.format(ind_base), [best_test_cls_loss, best_test_cls_acc])
    np.save('res/FasterRCNN{}_test_reg.npy'.format(ind_base), [best_test_reg_loss, best_test_reg_acc])

    np.save('res/FasterRCNN{}_train_rcnn_loss.npy'.format(ind_base), list_loss_rcnn[:epoch, :])
    np.save('res/FasterRCNN{}_train_rcnn_acc.npy'.format(ind_base), list_acc_rcnn[:epoch, :])
    np.save('res/FasterRCNN{}_train_cls_loss.npy'.format(ind_base), list_loss_cls[:epoch, :])
    np.save('res/FasterRCNN{}_train_cls_acc.npy'.format(ind_base), list_acc_cls[:epoch, :])
    np.save('res/FasterRCNN{}_train_reg_loss.npy'.format(ind_base), list_loss_reg[:epoch, :])
    np.save('res/FasterRCNN{}_train_reg_acc.npy'.format(ind_base), list_acc_reg[:epoch, :])

  return time_count


# main code
def main(ind_base):
  dict_train = dataProcess(readTest=False)
  dict_test = dataProcess(readTest=True)
  
  timeCost = trainAndTest(dict_train, dict_test, ind_base)
  print '\n=========== The training process has completed! Total [{} epochs] using [time {} s] ============'.format(iteration, timeCost)



if __name__ == "__main__":
  '''
    ind_base refers to different basenet structure
    - 0: ConvNet Block
    - 1: MobileNet Block
    - 2: ResNet Block
  '''
  ind_base = 2
  main(ind_base)

  '''
    Plot function
  '''
  processPlot(ind_base)