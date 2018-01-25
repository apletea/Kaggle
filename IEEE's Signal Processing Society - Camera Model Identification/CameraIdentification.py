import pandas as pd
import numpy as np
import tensorflow as tf
import keras

config = tf.ConfigProto( device_count = {'GPU' : 1, 'CPU':16})
sess = tf.Session(config=config)
keras.backend.set_session(sess)

import random
import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm
from keras.layers import *
from keras.models import *
from keras.callbacks import ModelCheckpoint
from keras.utils.np_utils import to_categorical


path_to_data = '/home/uberuser/datasets/signal/'

cameras = pd.read_csv(path_to_data + 'cameras.txt')
print cameras
back_map = {i:cameras['camera'][i] for i in xrange(0,len(cameras))}
map_back = {cameras['camera'][i]: i for i in xrange(0,len(cameras))}
data = pd.read_csv(path_to_data + 'sample_submission.csv')

from keras.layers import *
from keras.models import *
from keras.applications import *
from keras.optimizers import *
from keras.regularizers import *
from keras.applications.inception_v3 import preprocess_input
from sklearn.metrics import log_loss
from scipy import signal
size = 75
def get_model():
    model = Xception(weights='imagenet',include_top=False,input_shape=(size, size, 3))
    for layer in model.layers:
        layer.trainable = False
    x = model.output
    x = Flatten()(x)
    x = Activation('selu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(10, activation='softmax')(x)
    return Model(inputs=model.input, outputs=predictions)

def get_my_model():
    inputs = Input((size,size,3))
    conv1 = Conv2D(64, (9, 9), padding='valid', activation='elu')(inputs)
    conv1 = BatchNormalization(momentum = 0.99)(conv1)
    pool1 = MaxPool2D(pool_size=(2, 2))(conv1)
    drop1 = Dropout(0.1)(pool1)

    conv2 = Conv2D(64, (5, 5), padding='valid', activation='elu')(drop1)
    conv2 = BatchNormalization(momentum = 0.95)(conv2)
    pool2 = AvgPool2D(pool_size=(2, 2))(conv2)
    drop2 = Dropout(0.1)(pool2)

    conv3 = Conv2D(64, (3, 3), padding='valid', activation='elu')(drop2)
    conv3 = BatchNormalization(momentum = 0.95)(conv3)
    pool3 = MaxPool2D(pool_size=(2, 2))(conv3)
    drop3 = Dropout(0.1)(pool3)

    conv4 = Conv2D(64, (3, 3), padding='valid', activation='elu')(drop3)
    pool4 = AvgPool2D(pool_size=(2, 2))(conv4)

    gp = GlobalMaxPooling2D()(pool4)

    out = Dense(10, activation = 'softmax')(gp)
    model = Model(inputs,out)
    return model

vgg16 = get_model()
vgg16.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
def split_img():
    return None

def load_train_data():
    X = np.array([],np.float32)
    X = []
    y = []
    for camera in cameras['camera']:
        print camera
        cam_imgs = pd.read_csv(path_to_data + 'train/' + camera + '.txt')
        for i in xrange(0,len(cam_imgs)):
            img = cv2.imread(path_to_data + 'train/' + camera + '/' + cam_imgs['img'][i])
#            print img.shape[0]
            l =int ((img.shape[0]-size) * random.random())
            h =int ((img.shape[1]-size) * random.random())
            
            img_to_apend = img[l:l+size, h:h+size]
                   
                    #np.append((img_to_apend))
            X.append(img_to_apend)
            y.append(map_back[camera])
            
    return X,y
X,y = load_train_data() 

y = to_categorical(y, 10)
print len(X)
print len(y)
print y.shape
X = np.array(X,np.float32)/255
vgg19 = get_my_model()

vgg19.compile(loss='categorical_crossentropy', optimizer='sgd',metrics=['accuracy'])
checkpoint_path ='/home/uberuser/weights/weights.hdf5'
checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss', verbose=0, period=1)
vgg19.fit(X,y,nb_epoch=150, batch_size = 20, verbose=1, callbacks=[checkpoint], shuffle=247)

ans = path_to_data + 'sample_submission.csv'
ans = pd.read_csv(ans)
ans.head(5)
model = load_model('model.hdf5')

def load_test_data():
    X = np.array([],np.float32)
    X = []
    y = []
    for camera in cameras['camera']:
        print camera
        cam_imgs = pd.read_csv(path_to_data + 'train/' + camera + '.txt')
        for i in xrange(0,len(cam_imgs)):
            img = cv2.imread(path_to_data + 'train/' + camera + '/' + cam_imgs['img'][i])
#            print img.shape[0]
            l = img.shape[0] / size
            h = img.shape[1] / size
        
            img_to_apend = img[0:size, 0:size]
                   
                    #np.append((img_to_apend))
            X.append(img_to_apend)
            y.append(map_back[camera])
            
    return X,y

def load_test(ans):
    X = []
    for i in xrange(0,len(ans)):
        img = cv2.imread(path_to_data + 'test/' + ans['fname'][i])
        img_to_apend = img[0:size, 0:size]
        X.append(img_to_apend)
    
    return X
X = load_test(ans)
X = np.array(X,np.float32)/255
Y = model.predict(X)

sample = path_to_data + 'sample_submission.csv'
sample = pd.read_csv(sample)
for i in xrange(0,2640):
    sample['camera'][i] = back_map[ans[i]]
print sample.head(5) 
sample.to_csv('ans.csv',index=None)
