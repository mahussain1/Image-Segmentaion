from tensorflow.keras import backend as K
import  numpy as np
def iou_coef(y_true, y_pred, smooth=1):
  intersection = K.sum(K.abs(y_true * y_pred),axis=-1)
  union = K.sum(y_true,axis=-1)+K.sum(y_pred,axis=-1)-intersection
  iou = K.mean((intersection + smooth) / (union + smooth),axis=-1)
  return iou
  
def dice_coef(y_true, y_pred, smooth=1):
  intersection = K.sum(y_true * y_pred,axis=-1)
  union = K.sum(y_true,axis=-1) + K.sum(y_pred,axis=-1)
  dice = K.mean((2. * intersection + smooth)/(union + smooth),axis=-1)
  return dice