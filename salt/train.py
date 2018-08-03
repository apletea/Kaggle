import cv2
import numpy as np
import random
import skimage
from skimage import data
from sklearn.model_selection import train_test_split
import pandas as pd
from matplotlib import pyplot as plt
import os
import glob
import tensorflow as tf
import keras

config = tf.ConfigProto( device_count = {'GPU' : 1, 'CPU':8})
sess = tf.Session(config=config)
keras.backend.set_session(sess)

from keras.layers import *
from keras.models import *
from keras.optimizers import *
from keras.applications import *
from keras.regularizers import *
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras.models import load_model
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint , EarlyStopping, ReduceLROnPlateau, TensorBoard
from keras.utils.np_utils import to_categorical
from keras import backend as K
from keras.models import load_model
from sklearn.metrics import roc_curve, auc, average_precision_score, precision_recall_curve
from sklearn.utils import class_weight
from keras.models import *
from keras.optimizers import *
from keras.applications import *

from keras.utils import plot_model

TRAIN_FOLDER = './train/'
TEST_FOLDER = './test/'
NAME_MODEL = 'Dense_201_gpu'


def focal_loss_fixed(y_true, y_pred):
  gamma=2.0
  alpha=0.25
  pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
  pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
  return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))

def dice_coef(y_true, y_pred):
  y_true_f = K.flatten(y_true)
  y_pred_f = K.flatten(y_pred)
  intersection = K.sum(y_true_f * y_pred_f)
  return (2.0 * intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1.0)


def jacard_coef(y_true, y_pred):
  y_true_f = K.flatten(y_true)
  y_pred_f = K.flatten(y_pred)
  intersection = K.sum(y_true_f * y_pred_f)
  return (intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + 1.0)


def jacard_coef_loss(y_true, y_pred):
  return -jacard_coef(y_true, y_pred)


def dice_coef_loss(y_true, y_pred):
  return -dice_coef(y_true, y_pred)


def double_conv_layer(x, size, dropout, batch_norm):
  if K.image_dim_ordering() == 'th':
    axis = 1
  else:
    axis = 3
  conv = Convolution2D(size, (3, 3), padding='same')(x)
  if batch_norm is True:
    conv = BatchNormalization(axis=axis)(conv)
  conv = Activation('relu')(conv)
  conv = Convolution2D(size, (3, 3), padding='same')(conv)
  if batch_norm is True:
    conv = BatchNormalization(axis=axis)(conv)
  conv = Activation('relu')(conv)
  if dropout > 0:
     conv = Dropout(dropout)(conv)
  return conv

def double_conv_conc(x, size, dropout, batch_norm):
  conv = BatchNormalization()(x)
  conv = Activation('relu')(conv)
  conv = Convolution2D(size, (3, 3), padding='same')(conv)  

def MiddleFlow(input, num_output):
  a = SeparableConv2D(num_output, 3, activation='relu', padding='same')(input)
  a = BatchNormalization()(a)
  a = SeparableConv2D(num_output, 3, activation='relu', padding='same')(input)
  a = BatchNormalization()(a)
  a = SeparableConv2D(num_output, 3, activation='relu', padding='same')(input)
  a = BatchNormalization()(a)
  out = Concatenate()([a,input])
  out = Convolution2D(num_output, 1, activation='relu', padding='same')(out)
  out = BatchNormalization()(out)
  return out

def EntryFlow(input, num_output):
  a = SeparableConv2D(num_output, 3, activation='relu', padding='same')(input)
  a = BatchNormalization()(a)
  a = SeparableConv2D(num_output, 3,activation='relu', padding='same')(a)
  a = BatchNormalization()(a)
  a = MaxPooling2D(2)(a)
  b = Convolution2D(num_output, 1, strides=2, activation='relu')(input)
  b = BatchNormalization()(b)
  out = Concatenate()([a,b])
  out = Convolution2D(num_output, 1, activation='relu', padding='same')(out)
  out = BatchNormalization()(out)
  return out

def get_model(input_shape=(112,112,3)):
  filters = 32
  dropout = False
  batch_norm = True
  img_input = Input(shape=input_shape)

  conv112 = double_conv_layer(img_input, filters, dropout, batch_norm)

  pool56 = EntryFlow(conv112, filters*2)
  conv56 = MiddleFlow(pool56, filters*2)
  conv56 = MiddleFlow(conv56, filters*2)
  conv56 = MiddleFlow(conv56, filters*2)  

  pool28 = EntryFlow(conv56,  filters*4)
  conv28 = MiddleFlow(pool28, filters*4)
  conv28 = MiddleFlow(conv28, filters*4) 
  conv28 = MiddleFlow(conv28, filters*4) 

  pool14 = EntryFlow(conv28,  filters*8)
  conv14 = MiddleFlow(pool14, filters*8)
  conv14 = MiddleFlow(conv14, filters*8) 
  conv14 = MiddleFlow(conv14, filters*8) 

  pool7 = EntryFlow(conv14,   filters*16)
  conv7 = MiddleFlow(pool7, filters*16) 
  conv7 = MiddleFlow(conv7, filters*16) 
  conv7 = MiddleFlow(conv7, filters*16)

  up14 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv14], axis=-1)
  up14 = Convolution2D(filters*8, (1, 1), activation='relu')(up14)
  conv14 = MiddleFlow(up14, filters*8)
  conv14 = MiddleFlow(conv14, filters*8) 
  conv14 = MiddleFlow(conv14, filters*8)

  up28 = concatenate([UpSampling2D(size=(2, 2))(conv14), conv28], axis=-1)
  up28 = Convolution2D(filters*4, (1, 1), activation='relu')(up28)
  conv28 = MiddleFlow(up28, filters*4)
  conv28 = MiddleFlow(conv28, filters*4) 
  conv28 = MiddleFlow(conv28, filters*4) 

  up56 = concatenate([UpSampling2D(size=(2, 2))(conv28), conv56], axis=-1)
  up56 = Convolution2D(filters*2, (1, 1), activation='relu')(up56)
  conv56 = MiddleFlow(up56, filters*2)
  conv56 = MiddleFlow(conv56, filters*2)
  conv56 = MiddleFlow(conv56, filters*2) 

  up112 = concatenate([UpSampling2D(size=(2, 2))(conv56), conv112], axis=-1)
  conv112 = Convolution2D(1, (1, 1), activation='sigmoid')(up112)
  
  model = Model(img_input,conv112)
  #model.compile(optimizer='adam', loss=focal_loss_fixed, metrics=['accuracy',dice_coef])
  model.compile(optimizer='adam', loss=dice_coef_loss, metrics=['accuracy',dice_coef])
  return model

def padd_image(image, mask):
  channels = 3
  ans = np.zeros((112,112,channels))
  ans[:101,:101] = image
  return ans


def get_train():
  X = []
  Y = []
  for image in os.listdir(TRAIN_FOLDER + 'images'):
    img = padd_image(cv2.imread(TRAIN_FOLDER + 'images/{}'.format(image)),False)
    mask = padd_image(cv2.imread(TRAIN_FOLDER + 'masks/{}'.format(image)),True)
    X.append(img)
    Y.append(mask)
  X = np.array(X, np.float32) / 255
  Y = np.array(Y, np.float32) / 255
  return X,Y
   

def get_test():
  return []

callbacks = [EarlyStopping(monitor='val_loss',
                           patience=8,
                           verbose=1,
                           min_delta=1e-4),
             ReduceLROnPlateau(monitor='val_loss',
                               factor=0.1,
                               patience=4,
                               verbose=1,
                               epsilon=1e-4),
             ModelCheckpoint(monitor='val_loss',
                             filepath='weights/{}.h5'.format(NAME_MODEL),
                             save_best_only=True,
                             save_weights_only=True)]

X,Y = get_train()
model = get_model()
#model = load_model('models/{}.h5'.format(NAME_MODEL))
print (model.summary())
plot_model(model, 'dragon.png')
model.fit(X,Y[:,:,:,:1], epochs=100, validation_split=0.1, callbacks=callbacks)
model.save('models/{}.h5'.format(NAME_MODEL))
