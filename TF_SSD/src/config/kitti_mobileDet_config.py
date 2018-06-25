

"""Model configuration for pascal dataset"""

import numpy as np
import math
from config import base_model_config

def kitti_mobileDet_config():
  """Specify the parameters to tune below."""
  
  mc                       = base_model_config('KITTI')

  mc.VERSION               = 'V1'

  #mc.VERSION               = 'V1_05'
  #mc.VERSION               = 'V1_025'
  
  mc.QuantVersion          = 'V1' 

  mc.bQuant                = False
  mc.bQuantWeights         = True
  mc.bQuantActivations     = True
  mc.FirstQuantWeightsBitW = 8
  mc.FirstQuantActBitW     = 8
  mc.MiddleQuantWeightsBitW = 8
  mc.MiddleQuantActBitW     = 6
  mc.LastQuantWeightsBitW  = 8
  mc.LastQuantActBitW      = 8
  mc.QuantWeightsBitW      = 8
  mc.QuantActBitW          = 8

  print ('mobiledet.VERSION:',mc.VERSION)
  if mc.bQuant == True:
    print ('======>MiddleQuantWeightsBitW:',mc.MiddleQuantWeightsBitW)
    print ('======>MiddleQuantActBitW:',mc.MiddleQuantActBitW)
    
  mc.INPUT_WIDTH_SCALE     = 1
  mc.INPUT_HEIGHT_SCALE    = 1

  mc.IMAGE_WIDTH           = int(1248 * mc.INPUT_WIDTH_SCALE)
  mc.IMAGE_HEIGHT          = int(384  * mc.INPUT_HEIGHT_SCALE)

  mc.BATCH_SIZE            = 12
  
  #mc.WEIGHT_DECAY          = 0.0001
  #===v0
  mc.WEIGHT_DECAY          = 0.00002 #0.00004 #0.00001
  mc.LEARNING_RATE         = 0.01
  mc.DECAY_STEPS           = 30000
  mc.LR_DECAY_FACTOR       = 0.5

  #===v1
  '''
  mc.WEIGHT_DECAY          = 0.0001
  mc.LEARNING_RATE         = 0.01
  mc.DECAY_STEPS           = 50000
  mc.LR_DECAY_FACTOR       = 0.2
  '''
  mc.ADD_WEIGHT_DECAY_TO_LOSS = True
  mc.BATCH_NORM_EPSILON   = 0.0001 #0.001
  mc.BATH_NORM_DECAY      = 0.99#0.9997
 
  mc.MAX_GRAD_NORM         = 1.0
  mc.MOMENTUM              = 0.9

  mc.B_FOCAL_LOSS          = False 
  mc.FOCAL_LOSS_APHA       = 0.25
  mc.FOCAL_LOSS_GAMMA      = 2 

  mc.B_CONF_LOSS_V1        = False
  mc.CONF_LOSS_V1_ALPHA    = 1.5
  mc.CONF_LOSS_V1_GAMMA    = 2
  mc.CONF_LOSS_V1_BETA     = 0.5
  
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

  mc.bDoreFa               = False
  mc.BITW                  = 6
  mc.BITA                  = 32
  mc.BITG                  = 32 
  mc.is_training           = False

  if  mc.bQuant == True:
    mc.VERSION = 'V1'
    
  return mc


def kitti_mobileDet_V1_025_config():
  """Specify the parameters to tune below."""

  mc                       = base_model_config('KITTI')

  mc.VERSION               = 'V1_025'
  
  mc.QuantVersion          = 'V1' 
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
  mc.LastQuantWeightsBitW  = 8
  mc.LastQuantActBitW      = 8
  mc.QuantWeightsBitW      = 8
  mc.QuantActBitW          = 8

  print ('mobiledet.VERSION:',mc.VERSION)
  if mc.bQuant == True:
    print ('======>MiddleQuantWeightsBitW:',mc.MiddleQuantWeightsBitW)
    print ('======>MiddleQuantActBitW:',mc.MiddleQuantActBitW)
    
  mc.INPUT_WIDTH_SCALE     = 1.0
  mc.INPUT_HEIGHT_SCALE    = 1.0

  mc.IMAGE_WIDTH           = int(1248 * mc.INPUT_WIDTH_SCALE)
  mc.IMAGE_HEIGHT          = int(384  * mc.INPUT_HEIGHT_SCALE)

  mc.BATCH_SIZE            = 32
  
  #mc.WEIGHT_DECAY         = 0.0001
  #===v0
  mc.WEIGHT_DECAY          = 0.00001 #0.00004 #0.00001
  mc.LEARNING_RATE         = 0.01
  mc.DECAY_STEPS           = 30000
  mc.LR_DECAY_FACTOR       = 0.5

  mc.QUEUE_CAPACITY        = 128

 
  #===v1
  mc.ACTIVATION_FUNC       = 'relu'#'relu6'
 
  '''
  mc.WEIGHT_DECAY          = 0.0001
  mc.LEARNING_RATE         = 0.01
  mc.DECAY_STEPS           = 50000
  mc.LR_DECAY_FACTOR       = 0.2
  '''
  
  mc.ADD_WEIGHT_DECAY_TO_LOSS = True
  mc.BATCH_NORM_EPSILON   = 0.0001 #0.001
  mc.BATH_NORM_DECAY      = 0.99#0.9997
 
  mc.MAX_GRAD_NORM         = 1.0
  mc.MOMENTUM              = 0.9

  mc.B_CONF_LOSS_V1        = True

  ##map 0.78 recall:0.813
  '''
  mc.CONF_LOSS_GAMMA       = 24
  mc.CONF_LOSS_X_OFFSET    = -0.26
  mc.CONF_LOSS_Y_OFFSET    = -0.12
  '''
  
  mc.CONF_LOSS_GAMMA       = 28
  mc.CONF_LOSS_X_OFFSET    = -0.26
  mc.CONF_LOSS_Y_OFFSET    = -0.
  
  mc.LOSS_COEF_BBOX        = 5.0
  mc.LOSS_COEF_CONF_POS    = 75.0
  mc.LOSS_COEF_CONF_NEG    = 100#300.0
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
  mc.ANCHOR_PER_GRID       = len(get_sapce_shape())

  mc.bDoreFa               = False
  mc.BITW                  = 6
  mc.BITA                  = 32
  mc.BITG                  = 32 
  #mc.is_training           = False

  if  mc.bQuant == True:
    mc.VERSION = 'V1'
    
  return mc


def get_sapce_shape():
  space_shape = np.array(
          [[  36.,  37.], [ 366., 174.], [ 115.,  59.],
           [ 162.,  87.], [  38.,  90.], [ 258., 173.],
           [ 224., 108.], [  78., 170.], [  72.,  43.]])
  return space_shape
  
'''
def get_sapce_shape():
  anchor_base = np.array([45.,90.,180.,360.])
  size_base = np.array([0.5,1.0,2.])

  anchor = np.zeros(shape=(len(anchor_base)*len(size_base),2)) 

  len_size = len(size_base)
  print (np.shape(anchor))
  for i in range(0,len(anchor_base)):
    for j in range(0,len(size_base)):
      tmp = ((anchor_base[i] * anchor_base[i]) / size_base[j]) ** 0.5
      anchor[i*len_size+j][0] = tmp 
      anchor[i*len_size+j][1] = tmp * size_base[j]

  return anchor
'''

def set_anchors(mc):

  space_shape = get_sapce_shape()
  H, W, B = 24, 78, len(space_shape)
  #H, W, B = 12, 39, 9
  H *= mc.INPUT_HEIGHT_SCALE
  W *= mc.INPUT_WIDTH_SCALE
  H = int(round(H))
  W = int(round(W))
  
  space_shape[:,0] *=  mc.INPUT_HEIGHT_SCALE
  space_shape[:,1] *=  mc.INPUT_WIDTH_SCALE
  
  anchor_shapes = np.reshape(
      [space_shape] * H * W,
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
  anchors = np.reshape(
      np.concatenate((center_x, center_y, anchor_shapes), axis=3),
      (-1, 4)
  )

  return anchors

