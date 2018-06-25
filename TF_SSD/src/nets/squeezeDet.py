# -*- coding: utf-8 -*-
# Author:  08/25/2016

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
from dorefa import get_dorefa

class SqueezeDet(ModelSkeleton):
  def __init__(self, mc, gpu_id=0):
    with tf.device('/gpu:{}'.format(gpu_id)):
      ModelSkeleton.__init__(self, mc)
      if mc.squeezeDet_version == 'V0':
        self._add_forward_graph()
      else:
        self._add_forward_graph_V1()
        
      self._add_interpretation_graph()
      self._add_loss_graph()
      self._add_train_graph()
      self._add_viz_graph()

  def _add_forward_graph_V1(self):
    """NN architecture."""
  
    mc = self.mc
    if mc.LOAD_PRETRAINED_MODEL:
      assert tf.gfile.Exists(mc.PRETRAINED_MODEL_PATH), \
          'Cannot find pretrained model at the given path:' \
          '  {}'.format(mc.PRETRAINED_MODEL_PATH)
      self.caffemodel_weight = joblib.load(mc.PRETRAINED_MODEL_PATH)

    if mc.bDoreFa == True:
      self.image_input /= 255.0
      mc.BITW                  = 32
      mc.BITA                  = 32
      mc.BITG                  = 32 
      
    if mc.bQuant == True:
      mc.bQuantWeights       = True
      mc.QuantWeightsBitW    = mc.FirstQuantWeightsBitW #
      mc.bQuantActivations   = False #因为输入已经是8位了

    '''
    if mc.bDoreFa == True:
      conv1 = self._conv_bn_layer(
          'conv1', self.image_input,'conv1','bn1','scale1',filters=64, size=3, stride=2,
          padding='SAME', freeze=True,conv_with_bias=True)
    else:
    '''
    self._add_act(self.image_input,'data')
    
    conv1 = self._conv_layer(
        'conv1', self.image_input, filters=64, size=3, stride=2,
        padding='SAME', freeze=True)
    self._add_act(conv1,'conv1')
    
    if mc.bDoreFa == True:
      mc.BITW                  = 4
      mc.BITA                  = 4
      mc.BITG                  = 32 
      
    if mc.bQuant == True:
      mc.bQuantWeights       = True
      mc.QuantWeightsBitW    = mc.MiddleQuantWeightsBitW
      mc.bQuantActivations   = True 
      mc.QuantActBitW        = mc.MiddleQuantActBitW

    
    pool1 = self._pooling_layer(
        'pool1', conv1, size=3, stride=2, padding='SAME')
    self._add_act(pool1,'pool1')
    
    fire2 = self._fire_layer(
        'fire2', pool1, s1x1=16, e1x1=64, e3x3=64, freeze=False)
    self._add_act(fire2,'fire2')
    
    fire3 = self._fire_layer(
        'fire3', fire2, s1x1=16, e1x1=64, e3x3=64, freeze=False)
    self._add_act(fire2,'fire3')
    
    pool3 = self._pooling_layer(
        'pool3', fire3, size=3, stride=2, padding='SAME')
    self._add_act(pool3,'pool3')
    
    fire4 = self._fire_layer(
        'fire4', pool3, s1x1=32, e1x1=128, e3x3=128, freeze=False)
    self._add_act(fire4,'fire4')
    
    fire5 = self._fire_layer(
        'fire5', fire4, s1x1=32, e1x1=128, e3x3=128, freeze=False)
    self._add_act(fire5,'fire5')
    
    pool5 = self._pooling_layer(
        'pool5', fire5, size=3, stride=2, padding='SAME')
    self._add_act(pool5,'pool5')
    
    fire6 = self._fire_layer(
        'fire6', pool5, s1x1=48, e1x1=192, e3x3=192, freeze=False)
    self._add_act(fire6,'fire6')
    
    fire7 = self._fire_layer(
        'fire7', fire6, s1x1=48, e1x1=192, e3x3=192, freeze=False)
    self._add_act(fire7,'fire7')
    
    fire8 = self._fire_layer(
        'fire8', fire7, s1x1=64, e1x1=256, e3x3=256, freeze=False)
    self._add_act(fire8,'fire8')
    
    fire9 = self._fire_layer(
        'fire9', fire8, s1x1=64, e1x1=256, e3x3=256, freeze=False)
    self._add_act(fire9,'fire9')
    
    # Two extra fire modules that are not trained before
    fire10 = self._fire_layer(
        'fire10', fire9, s1x1=96, e1x1=384, e3x3=384, freeze=False)
    self._add_act(fire10,'fire10')
    
    fire11 = self._fire_layer(
        'fire11', fire10, s1x1=96, e1x1=384, e3x3=384, freeze=False)
    self._add_act(fire11,'fire11')
    
    '''
    if mc.bQuant == True:
       fire12 = self._fire_layer(
        'fire12', fire11, s1x1=144, e1x1=512, e3x3=512, freeze=False)
       fire13 = self._fire_layer(
        'fire13', fire12, s1x1=144, e1x1=512, e3x3=512, freeze=False)
    '''

    if mc.is_training == False:
      dropout11 = tf.nn.dropout(fire11, 1.0, name='drop11')
    else:
      if mc.bDoreFa == True:
          dropout11 = tf.nn.dropout(fire11, 1.0, name='drop11')
      elif mc.bQuant == True:
          dropout11 = tf.nn.dropout(fire11, 0.8, name='drop11')
      else:
          dropout11 = tf.nn.dropout(fire11, 0.5, name='drop11')
    
    if mc.bQuant == True:
      mc.bQuantWeights       = True
      mc.QuantWeightsBitW    = mc.LastQuantWeightsBitW
      mc.bQuantActivations   = True
      mc.QuantActBitW        = mc.LastQuantActBitW

    if mc.bDoreFa == True:
      mc.BITW                  = 32
      mc.BITA                  = 32
      mc.BITG                  = 32 

    self._add_act(dropout11,'dropout11')
    num_output = mc.ANCHOR_PER_GRID * (mc.CLASSES + 1 + 4)
    
    self.preds = self._conv_layer(
        'conv12', dropout11, filters=num_output, size=3, stride=1,
        padding='SAME', xavier=False, relu=False, stddev=0.0001)
        
    self._add_act(self.preds,'conv12')
    
  def _add_forward_graph(self):
    """NN architecture."""
    mc = self.mc
    if mc.LOAD_PRETRAINED_MODEL:
      assert tf.gfile.Exists(mc.PRETRAINED_MODEL_PATH), \
          'Cannot find pretrained model at the given path:' \
          '  {}'.format(mc.PRETRAINED_MODEL_PATH)
      self.caffemodel_weight = joblib.load(mc.PRETRAINED_MODEL_PATH)
      #print ("------------------------------load pretain model:",mc.PRETRAINED_MODEL_PATH)

    print ('====>_add_forward_graph')
    if mc.bDoreFa == True:
      self.image_input /= 255.0
      mc.BITW                  = 32
      mc.BITA                  = 32
      mc.BITG                  = 32 
    
    if mc.bQuant == True:
      mc.bQuantWeights       = True
      mc.QuantWeightsBitW    = 10
      mc.bQuantActivations   = False #因为输入已经是8位了
      #mc.QuantActBitW        = 8
      
    conv1 = self._conv_layer(
        'conv1', self.image_input, filters=64, size=3, stride=2,
        padding='VALID', freeze=True)
    self._add_act(conv1,'conv1')
    
    pool1 = self._pooling_layer(
        'pool1', conv1, size=3, stride=2, padding='VALID')

    if mc.bQuant == True:
       mc.BITW                  = 1
       mc.BITA                  = 4
       mc.BITG                  = 32 
    
    '''
    # self._add_act(conv1,'conv1')
    fw, fa, fg = get_dorefa(mc.BITW,mc.BITA,mc.BITG)
    if mc.bDoreFa == True:
       print ("------------------------------use dorefa to clip activations")
       pool1 = tf.clip_by_value(pool1, 0.0, 1.0)
       #pool1 = pool1 / tf.reduce_max(pool1);
       pool1 = fa(pool1)
    '''
    
    # self._add_act(pool1,'pool1')
    
    fire2 = self._fire_layer(
        'fire2', pool1, s1x1=16, e1x1=64, e3x3=64, freeze=False)
    self._add_act(fire2,'fire2')
    
    fire3 = self._fire_layer(
        'fire3', fire2, s1x1=16, e1x1=64, e3x3=64, freeze=False)
    self._add_act(fire3,'fire3')

    #print ("fire3:",fire3)
    pool3 = self._pooling_layer(
        'pool3', fire3, size=3, stride=2, padding='VALID')
    self._add_act(pool3,'pool3')
    
    fire4 = self._fire_layer(
        'fire4', pool3, s1x1=32, e1x1=128, e3x3=128, freeze=False)
    self._add_act(fire4,'fire4')
    
    fire5 = self._fire_layer(
        'fire5', fire4, s1x1=32, e1x1=128, e3x3=128, freeze=False)
    self._add_act(fire4,'fire4')
    
    pool5 = self._pooling_layer(
        'pool5', fire5, size=3, stride=2, padding='VALID')
    self._add_act(pool5,'pool5')
    
    fire6 = self._fire_layer(
        'fire6', pool5, s1x1=48, e1x1=192, e3x3=192, freeze=False)
    fire7 = self._fire_layer(
        'fire7', fire6, s1x1=48, e1x1=192, e3x3=192, freeze=False)
    fire8 = self._fire_layer(
        'fire8', fire7, s1x1=64, e1x1=256, e3x3=256, freeze=False)
    fire9 = self._fire_layer(
        'fire9', fire8, s1x1=64, e1x1=256, e3x3=256, freeze=False)
    #self._add_act(fire9,'fire9-concat')
    
    # Two extra fire modules that are not trained before
    fire10 = self._fire_layer(
        'fire10', fire9, s1x1=96, e1x1=384, e3x3=384, freeze=False)
    fire11 = self._fire_layer(
        'fire11', fire10, s1x1=96, e1x1=384, e3x3=384, freeze=False)

    self._add_act(fire11,'fire11-concat')
    
    if mc.bQuant == True:
      mc.bQuantWeights       = True
      mc.QuantWeightsBitW    = 10
      mc.bQuantActivations   = True
      mc.QuantActBitW        = 8
      
    if mc.bDoreFa == True:
      mc.BITW                  = 32
      mc.BITA                  = 32
      mc.BITG                  = 32 
      
    if mc.bDoreFa == True or mc.bQuant == True:
        dropout11 = tf.nn.dropout(fire11, 1.0, name='drop11')
    else:
        dropout11 = tf.nn.dropout(fire11, self.keep_prob, name='drop11')
    self._add_act(dropout11,'dropout11')
    
    num_output = mc.ANCHOR_PER_GRID * (mc.CLASSES + 1 + 4)
    print ("num_output:",num_output)

        
    self.preds = self._conv_layer(
        'conv12', dropout11, filters=num_output, size=3, stride=1,
        padding='SAME', xavier=False, relu=False, stddev=0.0001)
    self._add_act(self.preds,'conv12')
    print ("self.preds:",self.preds) 
  
  def _fire_layer(self, layer_name, inputs, s1x1, e1x1, e3x3, stddev=0.01,
      freeze=False):
    """Fire layer constructor.

    Args:
      layer_name: layer name
      inputs: input tensor
      s1x1: number of 1x1 filters in squeeze layer.
      e1x1: number of 1x1 filters in expand layer.
      e3x3: number of 3x3 filters in expand layer.
      freeze: if true, do not train parameters in this layer.
    Returns:
      fire layer operation.
    """
    
    mc = self.mc

    sq1x1 = self._conv_layer(
        layer_name+'/squeeze1x1', inputs, filters=s1x1, size=1, stride=1,
        padding='SAME', stddev=stddev, freeze=freeze)
    self._add_act(sq1x1,layer_name+'-squeeze1x1')
    
    ex1x1 = self._conv_layer(
        layer_name+'/expand1x1', sq1x1, filters=e1x1, size=1, stride=1,
        padding='SAME', stddev=stddev, freeze=freeze)
    self._add_act(ex1x1,layer_name+'-expand1x1')
    
    ex3x3 = self._conv_layer(
        layer_name+'/expand3x3', sq1x1, filters=e3x3, size=3, stride=1,
        padding='SAME', stddev=stddev, freeze=freeze)
    self._add_act(ex3x3,layer_name+'-expand3x3')
    
    return tf.concat([ex1x1, ex3x3], 3, name=layer_name+'/concat')
