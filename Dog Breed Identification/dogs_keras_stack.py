%matplotlib inline
import cv2
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import skimage
from skimage import data
from matplotlib import pyplot as plt
import keras
import random


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

im_size = 227 
chanels = 3
def get_model():
    target_in = Input(shape=(im_size,im_size,chanels))
    img_in    = Input(shape=(im_size,im_size,chanels))
    model_a = ResNet50(include_top=False,weights=None)
    model_b = Xception(include_top=False,weights=None)
    target_features = model_a(target_in)
    target_pool = GlobalAveragePooling2D()(target_features) 
    img_features    = model_b(img_in)
    img_pool = GlobalAveragePooling2D()(img_features)
    tmp = concatenate([target_pool,img_pool],axis=-1)
    tmp = Dense(256)(tmp)
    tmp = Dense(4)(tmp)
    model = Model([target_in,img_in],tmp)
    return model

model = get_model()
model.summary()
plot_model(model, to_file='model.png')
