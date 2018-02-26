import numpy as np
import cv2
import pandas as pd
import os
import tensorflow as tf
import keras

config = tf.ConfigProto( device_count = {'GPU' : 1, 'CPU':4})
sess = tf.Session(config=config)
keras.backend.set_session(sess)

import xgboost as xgb
from keras.layers import *
from keras.models import *
from keras.applications import *
from keras.optimizers import *
from keras.regularizers import *
from keras.applications.inception_v3 import preprocess_input
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import StratifiedKFold
from keras.utils.np_utils import to_categorical


def get_features(MODEL,imgs):
    model = MODEL(include_top=False, input_shape=(im_size, im_size, 3), weights='imagenet')

    inputs = Input((im_size,im_size,3))
    x = model.output
    x = GlobalAveragePooling2D()(x)
    res_model = Model(inputs=model.input, outputs=x)
    features = res_model.predict(imgs,batch_size=64,verbose=1)
    return features

im_size = 224
df_train = '/home/please/work/dogs/Images/'
df_test = '/home/please/work/dogs/test/'

breeds = os.listdir(df_train)

breed2int = {breeds[i] : i for i in xrange(0,len(breeds))}
int2breed = {i : breeds[i] for i in xrange(0,len(breeds))}

X = []
y = []
for dir in os.listdir(df_train):
    for img in os.listdir(df_train+dir):
        im = cv2.resize(cv2.imread(df_train + dir + '/' + img),(im_size,im_size))
        X.append(df_train + dir + '/' + img)
        y.append(breed2int[dir])
    
X_im = np.array([cv2.resize(cv2.imread(img),(im_size,im_size)) for img in X],np.float32) / 255    
y = to_categorical(y)

vgg_features = get_features(VGG19,X_im)

xg_train = xgb.DMatrix(vgg_features, y)
param = {}
# use softmax multi-class classification
param['objective'] = 'multi:softmax'
# scale weight of positive examples
param['eta'] = 0.1
param['max_depth'] = 10
param['silent'] = 1
param['nthread'] = 4
param['num_class'] = 120

bst = xgb.train(param,xg_train,5)

sample = pd.read_csv('/home/please/work/dogs/sample_submission.csv')
test = []
for file in sample['id']:
    test.append(cv2.resize(cv2.imread('test/{}.jpg'.format(file)),(im_size,im_size)))
test = np.array(test,np.float32) / 255
test_features = get_features(VGG19,test)
pred = bst.predict(test_features)
