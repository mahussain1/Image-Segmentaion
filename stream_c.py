import os
import time
import numpy as np
import cv2
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from config import imshape, model_name, n_classes
from utils_test import add_masks
from models import dice
import Metrics

# Eager execution provides an imperative interface to TensorFlow. With eager execution enabled, TensorFlow functions execute operations immediately 
# and return concrete values


tf.enable_eager_execution()
MODE = 'argmax'
model = load_model(os.path.join('models', model_name+'.model'),
                   custom_objects={'dice': dice})              


vs = cv2.imread('./test/4.jpg')
print(vs.shape)
cv2.imshow('Original Image',vs)
cv2.waitKey(1)
time.sleep(2)

_mask = cv2.imread('mask/4.png', 1)
grayImage = cv2.cvtColor(_mask, cv2.COLOR_BGR2GRAY)
(thresh, mask) = cv2.threshold(grayImage, 127, 255, cv2.THRESH_BINARY)
cv2.imshow('Ground Truth', mask)
cv2.waitKey(1)
time.sleep(2)



tmp = np.expand_dims(vs, axis=0)
roi_pred = model.predict(tmp)


if MODE == 'argmax':
    if n_classes == 1:
        roi_pred = roi_pred.squeeze()
        roi_softmax = np.stack([1-roi_pred, roi_pred], axis=2)
        roi_max = np.argmax(roi_softmax, axis=2)
        roi_pred = np.array(roi_max, dtype=np.float32)
    elif n_classes > 1:
        roi_max = np.argmax(roi_pred.squeeze(), axis=2)
        roi_pred = to_categorical(roi_max)

        
roi_mask = roi_pred.squeeze()*255.0
roi_mask = cv2.cvtColor(roi_mask, cv2.COLOR_GRAY2RGB)
cv2.imshow('Perdicted Mask', roi_mask)
cv2.waitKey(1) 
time.sleep(2)

cv2.imwrite('./pred/4.png', roi_mask)

grayImage = cv2.cvtColor(roi_mask, cv2.COLOR_BGR2GRAY)
(thresh, y_pred) = cv2.threshold(grayImage, 127, 255, cv2.THRESH_BINARY)

for i in range(256):
    for j in range(256):
        if mask[i,j]==255:
            mask[i,j] = 1
        else:
            mask[i,j] = mask[i,j]  
            
for i in range(256):
    for j in range(256):
        if y_pred[i,j]==255:
            y_pred[i,j] = 1
        else:
            y_pred[i,j] = y_pred[i,j]  
            
            
y_pred = tf.dtypes.cast(y_pred, tf.float32)
mask = tf.dtypes.cast(mask, tf.float32)



print(type(y_pred))
print(type(mask))

print('-------------------------------------------------------')
print('y_pred',y_pred)
print('mask',mask)
print('-------------------------------------------------------')

print(np.shape(y_pred))
print(np.shape(mask))

#intersection = np.logical_and(mask, y_pred)
#union = np.logical_or(mask, y_pred)
#iou_score = np.sum(intersection) / np.sum(union)

#print('Intersection over Union Score is ', iou_score)
#test_array = y_pred*mask

#print('shape of test array: ' ,np.shape(test_array))
#print('K summ: ',K.sum(test_array))

print('IOU Score by tf library:', Metrics.iou_coef(mask, y_pred))
print('Dice Score by tf library:', Metrics.dice_coef(mask, y_pred))
print('-------------------------------------------------------')