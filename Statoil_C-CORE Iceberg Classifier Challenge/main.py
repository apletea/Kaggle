import matplotlib.pyplot as plt
from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, Activation
from keras.models import Model, load_model
from keras.applications.vgg19 import preprocess_input,decode_predictions
import numpy as np
from scipy import signal
from keras.optimizers import SGD
from keras_tqdm import TQDMNotebookCallback
from keras.callbacks import ModelCheckpoint, EarlyStopping
from tqdm import tqdm_notebook
from keras.preprocessing.image import ImageDataGenerator
import xgboost
import os
import pandas
import cv2
from keras.layers import *
from keras.models import *
import pandas as pd
from tqdm import tqdm
import cv2


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

im_size = 75

data = pd.read_json('train.json')
print len(data)
X = []
y = []
data.inc_angle = data.inc_angle.apply(lambda x: -1 if x == 'na' else x)
for i in xrange(0,len(data)):
    img = cv2.imread('./band_res/{}.jpg'.format(data.id[i]))
    if data.inc_angle[i] != -1:
        X.append(img)
        y.append(data.is_iceberg[i])
print y
y = np.transpose(y,axes=-1)
print y
model = Sequential()
model.add(Convolution2D(50,11,4,input_shape=(75,75,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D((3,3),2))
model.add(Convolution2D(100,5,1))
model.add(Activation('relu'))
model.add(MaxPooling2D((3,3),2))
model.add(Convolution2D(200,3,1))
model.add(Activation('relu'))
model.add(MaxPooling2D((3,3),2))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(100))
model.add(Activation('selu'))
model.add(Dropout(0.5))
model.add(Dense(200))
model.add(Activation('selu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam')

X = np.array(X,np.float32)/255

print model.summary()
model.fit(X, y, nb_epoch=100, validation_split=0.1)
model.save('backup.h5')
