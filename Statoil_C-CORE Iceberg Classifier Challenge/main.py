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
from sklearn.metrics import log_loss
from scipy import signal

def get_VGG19():

    model = VGG19(weights='imagenet',include_top=False,input_shape=(75, 75, 3))
    for layer in model.layers:
        layer.trainable = False
    x = model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='selu')(x)
    x = Dropout(0.5)(x)
    x = Dense(64)(x)
    x = Activation('selu')(x)
    x = Dropout(0.5)(x)

    predictions = Dense(1, activation='sigmoid')(x)

    return Model(inputs=model.input, outputs=predictions)

def get_VGG16():
    model = VGG16(weights='imagenet', include_top=False, input_shape=(75, 75, 3))

    for layer in model.layers:
        layer.trainable = False
    x = model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='selu')(x)
    x = Dropout(0.5)(x)
    x = Dense(64)(x)
    x = Activation('selu')(x)
    x = Dropout(0.5)(x)

    predictions = Dense(1, activation='sigmoid')(x)

    return Model(inputs=model.input, outputs=predictions)

def get_Xception():
    model = Xception(weights='imagenet', include_top=False, input_shape=(75, 75, 3))
    for layer in model.layers:
        layer.trainable = False
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

def get_features(MODEL, data):
    model = MODEL(include_top=False, input_shape=(im_size, im_size, 3), weights='imagenet')

    inputs = Input((im_size,im_size,3))
    x = inputs
    x = Lambda(preprocess_input, name='preprocessing')(x)
    x = model(x)
    x = MaxPooling2D((2,2),1)(x)
    x = Flatten()(x)
    model = Model(inputs,x)

    features = model.predict(data,batch_size=64,verbose=1)
    features = np.concatenate([features,angle],axis=-1)
    print features
    print features.shape
    return features
from numpy import log
eps=1e-15
def logloss(true_label, predicted):
  p = np.clip(predicted, eps, 1 - eps)
  if true_label == 1:
    return -log(p)
  else:
    return -log(1 - p)

im_size = 75

data = pd.read_json('train.json')
print len(data)
X = []
y = []
X_band_1=np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in data["band_1"]])
X_band_2=np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in data["band_2"]])
angle = data.inc_angle
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
angle = []
for i in xrange(0,len(data)):
    img = imgs[i]
    if data.inc_angle[i] != -1:
        for j in xrange(0,15):
            X.append(img)
            y.append(data.is_iceberg[i])
            img = augment_image(img,i)
            angle.append([data.inc_angle[i]])

y = np.transpose(y,axes=-1)

X = np.array(X,np.float32)/255
angle = np.array(angle,np.float32)/90

# vgg16_features = get_features(VGG16,X)
# vgg19_features = get_features(VGG19,X)
# xception_features = get_features(Xception,X)
#
# X_train = np.concatenate([vgg16_features,vgg19_features,xception_features],axis=-1)

# inputs = Input(X_train.shape[1:])
# x = inputs
# x = Dropout(0.5)(x)
#
# x = Dense(1, activation='sigmoid')(x)
# model = Model(inputs,x)
# model.compile(optimizer='adam',
#               loss = 'binary_crossentropy')
# h = model.fit(X_train,y,batch_size=128,nb_epoch=1000,validation_split=0.05)
# model = get_VGG16()
# model.compile(optimizer='adam',loss = 'binary_crossentropy')
# model.fit(X,y,batch_size=128,nb_epoch=1000,validation_split=0.05)
# model.save('orig.h5')

# model_2 = Sequential()
# model_2.add(Dense(1, input_shape=[1], name='dense'))
# model = Sequential()
# model.add(Convolution2D(20,11,4,input_shape=(75,75,3)))
# model.add(Activation('relu'))
# model.add(AveragePooling2D((3,3),1))
# model.add(Convolution2D(40,7,1))
# model.add(Activation('relu'))
# model.add(AveragePooling2D((3,3),1))
# model.add(Convolution2D(80,5,1))
# model.add(Activation('relu'))
# model.add(MaxPooling2D((10,10),6))
# model.add(Flatten())
# model.add(Dropout(0.5))
# model.add(Dense(50))
# model.add(Activation('selu'))
# model.add(Dropout(0.5))
# model.add(Dense(50))
# model.add(Activation('selu'))
# model.add(Dropout(0.5,name='drop'))
# x = model.get_layer('drop').output
# x2 = model_2.get_layer('dense').output
# merge_one = concatenate([x, x2])
# merge_one = Dense(1, activation='sigmoid')(merge_one)
# model = Model(input=[model.input, model_2.input], output=merge_one)
# model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['binary_accuracy'])
# X = np.array(X,np.float32)/255
model = load_model('backup3.h5')
# y = model.predict(X)
# print y
# ans = pd.read_csv('sample_submission.csv')
# ans['is_iceberg'] = y
# ans.to_csv('ans,csv',index=None)
print model.summary()
for i in xrange(0,10):
    model.fit([X,angle], y, nb_epoch=120, batch_size=32, validation_split=0.15, shuffle=247)
    model.save('backup4.h5')
