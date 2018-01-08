import pandas as pd
import numpy as np
import tensorflow as tf
import keras

config = tf.ConfigProto( device_count = {'GPU' : 1, 'CPU':16})
sess = tf.Session(config=config)
keras.backend.set_session(sess)

from tqdm import tqdm
import cv2

from keras.layers import *
from keras.models import *
from keras.applications import *
from keras.optimizers import *
from keras.regularizers import *
from keras.applications.inception_v3 import preprocess_input
from scipy import signal

def get_VGG19():

    model = VGG19(weights='imagenet',include_top=False,input_shape=(75, 75, 3))

    x = model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='selu')(x)
    x = Dropout(0.5)(x)
    x = Dense(64)(x)
    x = Activation('selu')(x)
    x = Dropout(0.5)(x)

    predictions = Dense(1, activation='sigmoid')(x)

    return Model(inputs=model.input, outputs=predictions)


def pop(self):
    '''Removes a layer instance on top of the layer stack.'''
    if not self.outputs:
        raise Exception('Sequential model cannot be popped: model is empty.')
    else:
        self.layers.pop()
        if not self.layers:
            self.outputs = []
            self.inbound_nodes = []
            self.outbound_nodes = []
        else:
            self.layers[-1].outbound_nodes = []
            self.outputs = [self.layers[-1].output]
        self.built = False


def pop_last_3_layers(model):
    pop(model)
    pop(model)
    pop(model)
    return model

def export_model(model,filepath):
    model.save(filepath)

def import_weight(model, filepath):
    model = load_model(filepath)
    return model


def calc_mean_std(X):
    mean, std = np.zeros((3,)), np.zeros((3,))

    for x in X:
        x = x / 255.
        x = x.reshape(-1, 3)
        mean += x.mean(axis=0)
        std += x.std(axis=0)

    return mean / len(X), std / len(X)

def preprocess_poster(x, train_mean, train_std):
    x = x / 255.
    x -= train_mean
    x /= train_std
    return x


def augment_image(img,i):
    matrix = cv2.getRotationMatrix2D((75/2,75/2),(i+1)*6,1)
    aut = cv2.warpAffine(img,matrix,(75,75))
    return aut

im_size = 75

data = pd.read_json('train.json')
print len(data)
X = []
y = []
X_band_1=np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in data["band_1"]])
X_band_2=np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in data["band_2"]])
xder = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
yder = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
X_band_3 = (X_band_1 + X_band_2)/2
for i in xrange(0,len(X_band_1)):
    arrx = signal.convolve2d((X_band_1[i]), xder, mode='valid')
    arry = signal.convolve2d((X_band_2[i]), yder, mode='valid')

    mat = (np.hypot(arrx,arry))
    mat = cv2.copyMakeBorder(mat,1,1,1,1,cv2.BORDER_CONSTANT)
    X_band_3[i] = mat


imgs =  np.concatenate([X_band_1[:, :, :, np.newaxis]
                          , X_band_2[:, :, :, np.newaxis]
                         , X_band_3[:, :, :, np.newaxis]], axis=-1)

data.inc_angle = data.inc_angle.apply(lambda x: -1 if x == 'na' else x)
for i in xrange(0,len(data)):
    img = imgs[i]
    if data.inc_angle[i] != -1:
        for j in xrange(0,15):
            X.append(img)
            y.append(data.is_iceberg[i])
            img = augment_image(img,i)

y = np.transpose(y,axes=-1)

model = Sequential()
model.add(Convolution2D(30,20,4,input_shape=(75,75,3)))
model.add(Activation('relu'))
model.add(AveragePooling2D((3,3),1))
model.add(Convolution2D(60,15,1))
model.add(Activation('relu'))
model.add(AveragePooling2D((3,3),1))
model.add(Convolution2D(120,7,1))
model.add(Activation('relu'))
model.add(AveragePooling2D((5,5),4))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(50))
model.add(Activation('selu'))
model.add(Dropout(0.5))
model.add(Dense(50))
model.add(Activation('selu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam')

X = np.array(X,np.float32)/255

print model.summary()
model.fit(X, y, nb_epoch=100, validation_split=0.1)
model.save('backup2.h5')
