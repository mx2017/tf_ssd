# -*- coding: utf-8 -*-
"""SqueezeDet model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import joblib
from utils import util
from easydict import EasyDict as edict
import numpy as np
import tensorflow as tf
from nn_skeleton import ModelSkeleton
from nets.mobilenet_base import  mobilenet_v1_base
from collections import namedtuple

slim = tf.contrib.slim

Conv = namedtuple('Conv', ['kernel', 'stride', 'depth'])
DepthSepConv = namedtuple('DepthSepConv', ['kernel', 'stride', 'depth'])
_CONV_DEFS_v1 = [
    Conv(kernel=[3, 3], stride=2, depth=32),
    DepthSepConv(kernel=[3, 3], stride=1, depth=64),
    DepthSepConv(kernel=[3, 3], stride=2, depth=128),
    DepthSepConv(kernel=[3, 3], stride=1, depth=128),
    DepthSepConv(kernel=[3, 3], stride=2, depth=256),
    DepthSepConv(kernel=[3, 3], stride=1, depth=256),
    DepthSepConv(kernel=[3, 3], stride=2, depth=512),
    DepthSepConv(kernel=[3, 3], stride=1, depth=512),
    DepthSepConv(kernel=[3, 3], stride=1, depth=512),
    DepthSepConv(kernel=[3, 3], stride=1, depth=512),
    DepthSepConv(kernel=[3, 3], stride=1, depth=512),
    DepthSepConv(kernel=[3, 3], stride=1, depth=512),
    DepthSepConv(kernel=[3, 3], stride=1, depth=1024),
    DepthSepConv(kernel=[3, 3], stride=1, depth=1024),
    DepthSepConv(kernel=[3, 3], stride=1, depth=1024),
    DepthSepConv(kernel=[3, 3], stride=1, depth=1024)
]

_CONV_DEFS_v2 = [
    Conv(kernel=[3, 3], stride=2, depth=32),
    DepthSepConv(kernel=[3, 3], stride=1, depth=64),
    DepthSepConv(kernel=[3, 3], stride=2, depth=128),
    DepthSepConv(kernel=[3, 3], stride=1, depth=128),
    DepthSepConv(kernel=[3, 3], stride=2, depth=256),
    DepthSepConv(kernel=[3, 3], stride=1, depth=256),
    DepthSepConv(kernel=[3, 3], stride=2, depth=512),
    DepthSepConv(kernel=[3, 3], stride=1, depth=512),
    DepthSepConv(kernel=[3, 3], stride=1, depth=512),
    DepthSepConv(kernel=[3, 3], stride=1, depth=512),
    DepthSepConv(kernel=[3, 3], stride=1, depth=512),
    DepthSepConv(kernel=[3, 3], stride=1, depth=512)
]


_CONV_DEFS_v1_05 = [
    Conv(kernel=[3, 3], stride=2, depth=16),
    DepthSepConv(kernel=[3, 3], stride=1, depth=32),
    DepthSepConv(kernel=[3, 3], stride=2, depth=64),
    DepthSepConv(kernel=[3, 3], stride=1, depth=64),
    DepthSepConv(kernel=[3, 3], stride=2, depth=128),
    DepthSepConv(kernel=[3, 3], stride=1, depth=128),
    DepthSepConv(kernel=[3, 3], stride=2, depth=256),
    DepthSepConv(kernel=[3, 3], stride=1, depth=256),
    DepthSepConv(kernel=[3, 3], stride=1, depth=256),
    DepthSepConv(kernel=[3, 3], stride=1, depth=256),
    DepthSepConv(kernel=[3, 3], stride=1, depth=256),
    DepthSepConv(kernel=[3, 3], stride=1, depth=256),
    DepthSepConv(kernel=[3, 3], stride=1, depth=512),
    DepthSepConv(kernel=[3, 3], stride=1, depth=512),
    DepthSepConv(kernel=[3, 3], stride=1, depth=512),
    DepthSepConv(kernel=[3, 3], stride=1, depth=512)
]

_CONV_DEFS_v1_025 = [
    Conv(kernel=[3, 3], stride=2, depth=8),
    DepthSepConv(kernel=[3, 3], stride=1, depth=16),
    DepthSepConv(kernel=[3, 3], stride=2, depth=32),
    DepthSepConv(kernel=[3, 3], stride=1, depth=32),
    DepthSepConv(kernel=[3, 3], stride=2, depth=64),
    DepthSepConv(kernel=[3, 3], stride=1, depth=64),
    DepthSepConv(kernel=[3, 3], stride=2, depth=128),
    DepthSepConv(kernel=[3, 3], stride=1, depth=128),
    DepthSepConv(kernel=[3, 3], stride=1, depth=128),
    DepthSepConv(kernel=[3, 3], stride=1, depth=128),
    DepthSepConv(kernel=[3, 3], stride=1, depth=128),
    DepthSepConv(kernel=[3, 3], stride=1, depth=128),
    DepthSepConv(kernel=[3, 3], stride=1, depth=256),
    DepthSepConv(kernel=[3, 3], stride=1, depth=256),
    DepthSepConv(kernel=[3, 3], stride=1, depth=256),
    DepthSepConv(kernel=[3, 3], stride=1, depth=256)
]


#train use V0 and deploy use V1
class MobileDet(ModelSkeleton):
  def __init__(self, mc,gpu_id=0):
    with tf.device('/gpu:{}'.format(gpu_id)):
      ModelSkeleton.__init__(self, mc)

      if mc.ACTIVATION_FUNC == 'relu6':
        self.activation_fn = tf.nn.relu6
        print ('=========>USE relu6')
      else:
        self.activation_fn = tf.nn.relu
        print ('=========>USE relu')
      
      if mc.VERSION == 'V0':
        self._add_forward_graph()
        self._add_interpretation_graph()
        if mc.is_training == True:
          self._add_loss_graph()
          self._add_train_graph_v1('weights')
          self._add_viz_graph()

      elif mc.VERSION == 'V1' or mc.VERSION == 'V1_025':
        self._add_forward_graph_v1()
        self._add_interpretation_graph()
        #if mc.is_training == True:
        self._add_loss_graph()
        self._add_train_graph_v1()  
        self._add_viz_graph()
          
  def _add_forward_graph(self):
    """NN architecture."""

    mc = self.mc

    batch_norm_params = {
      'is_training': mc.is_training,
      'center': True,
      'scale': True,
      'decay': mc.BATH_NORM_DECAY, # 0.9997
      'epsilon': mc.BATCH_NORM_EPSILON, #0.001
    }

    regularize_depthwise = True
    # Set weight_decay for weights in Conv and DepthSepConv layers.
    weights_init = tf.truncated_normal_initializer(stddev=0.001)
    regularizer = tf.contrib.layers.l2_regularizer(mc.WEIGHT_DECAY)
    if regularize_depthwise:
        depthwise_regularizer = regularizer
    else:
        depthwise_regularizer = None
    
    final_endpoint = 'Conv2d_' + str(len(_CONV_DEFS_v1)-1)
    with slim.arg_scope([slim.conv2d, slim.separable_conv2d],
                      weights_initializer=weights_init,
                      activation_fn=tf.nn.relu6, normalizer_fn=slim.batch_norm):
      with slim.arg_scope([slim.batch_norm], **batch_norm_params):
        with slim.arg_scope([slim.conv2d], weights_regularizer=regularizer):
          with slim.arg_scope([slim.separable_conv2d],
                              weights_regularizer=depthwise_regularizer) as sc:
              net, end_points = mobilenet_v1_base(mc,self.image_input, scope='MobilenetV1',
                  min_depth=8,final_endpoint=final_endpoint,
                      depth_multiplier=1.0,
                          conv_defs=None)                       

    for key, value in end_points.items():
      self._add_act(value,key)
              
    #there exists batchnorm,so there is no need dropout layer.
    dropout11 = tf.nn.dropout(net, 1.0, name='drop11')

    num_output = mc.ANCHOR_PER_GRID * (mc.CLASSES + 1 + 4)

    if mc.ADD_WEIGHT_DECAY_TO_LOSS == True:
      self.preds = slim.conv2d(dropout11, num_output, [3, 3], stride=1,activation_fn=None,
                             normalizer_fn=None, scope='conv12',padding='SAME')
    else:
      self.preds = self._conv_layer(
          'conv12', dropout11, filters=num_output, size=3, stride=1,
          padding='SAME', xavier=False, relu=False, stddev=0.0001)
    print ("=====>self.preds:",self.preds)

  
  def my_mobilenet_v1_base(self,inputs,
                        min_depth=8,
                        depth_multiplier=1.0,
                        conv_defs=None,
                        scope=None):
    """Mobilenet v1."""
    
    depth = lambda d: max(int(d * depth_multiplier), min_depth)
    end_points = {}

    # Used to find thinned depths for each layer.
    if depth_multiplier <= 0:
      raise ValueError('depth_multiplier is not greater than zero.')

    mc = self.mc
    
    if conv_defs is None:
      if mc.VERSION == 'V1':
        conv_defs = _CONV_DEFS_v1

      elif mc.VERSION == 'V1_05':
        conv_defs = _CONV_DEFS_v1_05

      elif mc.VERSION == 'V1_025':
        conv_defs = _CONV_DEFS_v1_025

    final_endpoint = 'Conv2d_' + str(len(conv_defs)-1)
    
    with tf.variable_scope(scope, 'MobilenetV1', [inputs]):

      net = inputs
      for i, conv_def in enumerate(conv_defs):

        if i == 0:
          if mc.bQuant == True:
            mc.bQuantWeights       = True
            mc.QuantWeightsBitW    = mc.FirstQuantWeightsBitW #
            
            if mc.FirstQuantActBitW != 8:
              mc.bQuantActivations = True
              mc.QuantActBitW      = mc.FirstQuantActBitW
            else:
              mc.bQuantActivations   = False
        else:
          if mc.bQuant == True:
            mc.bQuantWeights       = True
            mc.QuantWeightsBitW    = mc.MiddleQuantWeightsBitW
            mc.bQuantActivations   = True 
            mc.QuantActBitW        = mc.MiddleQuantActBitW
        
        end_point_base = 'Conv2d_%d' % i
        
        layer_stride = conv_def.stride

        if isinstance(conv_def, Conv):
          end_point = end_point_base
          if i == 0:
            trainable = False           
          else:
            trainable = True   

          #trainable = True
          assert conv_def.kernel[0] == conv_def.kernel[1]
          net = self._conv_bn_layer(end_point,net,filters=depth(conv_def.depth),kernel_name='weights',
                                  relu=True,activation_fn=self.activation_fn,freeze=(not trainable),
                                        size=conv_def.kernel[0],stride=conv_def.stride) 
          #net = self._batch_norm_layer(net,end_point+'/BatchNorm')
          #net = slim.batch_norm(net, scope=end_point+'/BatchNorm')
          self._add_act(net,end_point)
          
          end_points[end_point] = net

          print ('end_point:',end_point)
          if end_point == final_endpoint:
            return net, end_points

        elif isinstance(conv_def, DepthSepConv):  
          end_point = end_point_base
          assert conv_def.kernel[0] == conv_def.kernel[1]
          net = self._depthwise_separable_conv(net,end_point_base,filters=conv_def.depth,
                        kernel_size=conv_def.kernel[0],stride=layer_stride,activation_fn=self.activation_fn)
          
          
          end_points[end_point] = net
          print ('end_point:',end_point)
          if end_point == final_endpoint:
            return net, end_points
        else:
          raise ValueError('Unknown convolution type %s for layer %d'
                           % (conv_def.ltype, i))
      raise ValueError('Unknown final endpoint %s' % final_endpoint)


  def _add_forward_graph_v1(self):
    mc = self.mc

    batch_norm_params = {
      'is_training': mc.is_training,
      'center': True,
      'scale': True,
      'decay': mc.BATH_NORM_DECAY, # 0.9997
      'epsilon': mc.BATCH_NORM_EPSILON, #0.001
    }
    self._add_act(self.image_input,'data')
    
    with slim.arg_scope([slim.batch_norm], **batch_norm_params):
      net, end_points = self.my_mobilenet_v1_base(self.image_input, scope='MobilenetV1',
                                                    min_depth=8,depth_multiplier=1.0,
                                                        conv_defs=None)                       
    #there exists batchnorm,so there is no need dropout layer.
    dropout11 = tf.nn.dropout(net, 1.0, name='drop11')

    num_output = mc.ANCHOR_PER_GRID * (mc.CLASSES + 1 + 4)

    if mc.bQuant == True:
      mc.bQuantWeights       = True
      mc.QuantWeightsBitW    = mc.LastQuantWeightsBitW
      mc.bQuantActivations   = True
      mc.QuantActBitW        = mc.LastQuantActBitW

    
    self.preds = self._conv_layer(
        'conv12', dropout11, filters=num_output, size=3, stride=1,
        padding='SAME', xavier=False, relu=False, stddev=0.0001,kernel_name='weights')


    #这里为了让exp函数进行查表优化，这里把prob和conf两部分先归一化到0到1之间,
    #后边使用数分类为整数和小数部分，分别查表就变成了乘法操作
   
    
    '''
    self.conv12_depthwise = self._conv_depthwise_layer(dropout11,'conv12_depthwise',num_output,
                                kernel_size=3,stride=1,
                                relu=True,activation_fn=tf.nn.relu,freeze=False)
                                
    self.preds = self._conv_layer(
            'conv12_pointwise', self.conv12_depthwise, filters=num_output, size=1, stride=1,
            padding='SAME', xavier=False, relu=False, stddev=0.0001,kernel_name='weights')
    '''   
    
    print ("=====>self.preds:",self.preds)
    self._add_act(self.preds,'conv12')

    