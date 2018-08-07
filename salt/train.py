import cv2
import numpy as np
import random
import skimage
from skimage import data
#from sklearn.model_selection import train_test_split
import pandas as pd
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
#from sklearn.metrics import roc_curve, auc, average_precision_score, precision_recall_curve
#from sklearn.utils import class_weight
from keras.models import *
from keras.optimizers import *
from keras.applications import *

from keras.utils import plot_model

TRAIN_FOLDER = './train/'
TEST_FOLDER = './test/'
NAME_MODEL = 'Unet_w_gpu'


depths = pd.read_csv('depths.csv', index_col='id')




def padd_image(image, mask):
  channels = 3
  ans = np.zeros((112,112,channels))
  ans[:101,:101] = image
  return ans


def get_train():
  X = []
  Y = []
  i = 0
  for image in os.listdir(TRAIN_FOLDER + 'images'):
    img = padd_image(cv2.imread(TRAIN_FOLDER + 'images/{}'.format(image)),False)
    mask = padd_image(cv2.imread(TRAIN_FOLDER + 'masks/{}'.format(image)),True)
    X.append(img)
    Y.append(mask)
    img = np.fliplr(img)
    mask = np.fliplr(mask)
    X.append(img)
    Y.append(mask)
    blur = cv2.blur(img,(5,5))
    X.append(blur)
    Y.append(mask)
    i+=1
  X = np.array(X, np.float32) / 255
  Y = np.array(Y, np.float32) / 255
  return  X,Y
   

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
model.fit(X,[Y[:,:,:,:1],Y[:,:,:,:1],Y[:,:,:,:1]], epochs=100, validation_split=0.05, callbacks=callbacks)
model.save('models/{}.h5'.format(NAME_MODEL))
model_k = Model(model.input, model.output[0])
model_k.save('models/inference_best.h5')
