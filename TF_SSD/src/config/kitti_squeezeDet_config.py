# Author:  08/25/2016

"""Model configuration for pascal dataset"""

import numpy as np

from config import base_model_config

debugs_values = []
debugs_values_name = []
def kitti_squeezeDet_config():
  """Specify the parameters to tune below."""

  mc                       = base_model_config('KITTI')
  mc.squeezeDet_version       = 'V1'
  mc.B_FOCAL_LOSS          = False 
  mc.FOCAL_LOSS_APHA       = 0.25
  mc.FOCAL_LOSS_GAMMA      = 2 
  
  mc.bQuant                = False
  mc.bQuantWeights         = True
  mc.bQuantActivations     = True
  mc.FirstQuantWeightsBitW = 8
  mc.FirstQuantActBitW     = 8
  mc.MiddleQuantWeightsBitW = 8
  mc.MiddleQuantActBitW     = 6
  mc.LastQuantWeightsBitW = 8
  mc.LastQuantActBitW     = 8

  if mc.bQuant == True:
    print ('======>MiddleQuantWeightsBitW:',mc.MiddleQuantWeightsBitW)
    print ('======>MiddleQuantActBitW:',mc.MiddleQuantActBitW)
  mc.QuantWeightsBitW      = 8
  mc.QuantActBitW          = 8
  
  mc.bDoreFa               = False
  mc.BITW                  = 4
  mc.BITA                  = 4
  mc.BITG                  = 32 

  if mc.squeezeDet_version == 'V0':
    mc.IMAGE_WIDTH           = 1242
    mc.IMAGE_HEIGHT          = 375
  else:
    mc.IMAGE_WIDTH           = 1248
    mc.IMAGE_HEIGHT          = 384
    
  mc.BATCH_SIZE            = 20#32


  mc.ADD_WEIGHT_DECAY_TO_LOSS = False
  
  if mc.bDoreFa == True:
    mc.WEIGHT_DECAY          = 0.00001 #0.00001
  elif mc.bQuant == True:
    mc.WEIGHT_DECAY          = 0.00001 #0.00001
  else:
    mc.WEIGHT_DECAY          = 0.0001 #0.0001
  
  mc.LEARNING_RATE         = 0.01

  if mc.bQuant == True:
    mc.DECAY_STEPS           = 15000
  else:
    mc.DECAY_STEPS           = 10000
    
  mc.LR_DECAY_FACTOR       = 0.5
  mc.MAX_GRAD_NORM         = 1.0
  mc.MOMENTUM              = 0.9
  
  mc.LOSS_COEF_BBOX        = 5.0
  mc.LOSS_COEF_CONF_POS    = 75.0
  mc.LOSS_COEF_CONF_NEG    = 100.0
  mc.LOSS_COEF_CLASS       = 1.0

  mc.PLOT_PROB_THRESH      = 0.4
  mc.NMS_THRESH            = 0.4
  mc.PROB_THRESH           = 0.005
  mc.TOP_N_DETECTION       = 64

  mc.DATA_AUGMENTATION     = True
  mc.DRIFT_X               = 150
  mc.DRIFT_Y               = 100
  mc.EXCLUDE_HARD_EXAMPLES = False

  mc.ANCHOR_BOX            = set_anchors(mc)
  mc.ANCHORS               = len(mc.ANCHOR_BOX)
  mc.ANCHOR_PER_GRID       = 9
 
  return mc

def get_debug_val():
    global debugs_values
    return debugs_values

def get_debug_val_name():
    global debugs_values_name
    return debugs_values_name
 
def set_anchors(mc):
  global debugs_values
  global debugs_values_name
  if mc.squeezeDet_version == 'V0':
    H, W, B = 22, 76, 9
  else:
    H, W, B = 24, 78, 9
  
  anchor_shapes = np.reshape(
      [np.array(
          [[  36.,  37.], [ 366., 174.], [ 115.,  59.],
           [ 162.,  87.], [  38.,  90.], [ 258., 173.],
           [ 224., 108.], [  78., 170.], [  72.,  43.]])] * H * W,
      (H, W, B, 2)
  )
  center_x = np.reshape(
      np.transpose(
          np.reshape(
              np.array([np.arange(1, W+1)*float(mc.IMAGE_WIDTH)/(W+1)]*H*B), 
              (B, H, W)
          ),
          (1, 2, 0)
      ),
      (H, W, B, 1)
  )
  
  #debugs_values += [center_x]
  #debugs_values_name += ['center_x']
  
  center_y = np.reshape(
      np.transpose(
          np.reshape(
              np.array([np.arange(1, H+1)*float(mc.IMAGE_HEIGHT)/(H+1)]*W*B),
              (B, W, H)
          ),
          (2, 1, 0)
      ),
      (H, W, B, 1)
  )
  #debugs_values += [center_y]
  #debugs_values_name += ['center_y']
  
  anchors = np.reshape(
      np.concatenate((center_x, center_y, anchor_shapes), axis=3),
      (-1, 4)
  )
  debugs_values += [anchors]
  debugs_values_name += ['anchors']
  
  return anchors
