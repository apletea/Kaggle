import pandas as pd
import numpy as np
import tensorflow as tf
import keras

config = tf.ConfigProto( device_count = {'GPU' : 1, 'CPU':16})
sess = tf.Session(config=config)
keras.backend.set_session(sess)

from keras.layers import *
from keras.models import *
from keras.applications import *

import cv2
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.svm import SVC

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

cameras = pd.read_csv('cameras.txt')
back_map = {i:cameras['camera'][i] for i in xrange(0,len(cameras))}

def load_train_data():
    X = []
    y = []
    return X,y

def load_test_data():
    X = []
    return X

data = pd.read_csv('sample_submission.csv')
for i in xrange(0,len(data)):
    img = cv2.imread('./test/'+data['fname'][i])
    print img.shape

X,y = load_train_data()


model = Sequential()
model.add(Convolution2D(96,30,1,input_shape=(64,64,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(2,2,2))
model.add(Convolution2D(256,20,1))
model.add(Activation('relu'))
model.add(Convolution2D(364,11,1))
model.add(Activation('relu'))
model.add(MaxPooling2D(3,3,3))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(50))
model.add(Activation('selu'))
model.add(Dropout(0.5))
model.add(Dense(50))
model.add(Activation('selu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

model.compile(loss='binary_crossentropy', optimizer='adam')

X = np.array(X,np.float32)/255

print model.summary()
model.fit(X, y, nb_epoch=200, batch_size=32, validation_split=0.05)
model.save('backup2.h5')

model.load('backup2.h5')
pop_last_3_layers(model)
X_features = model.predict(X)
clasifier = XGBClassifier()
clasifier.fit(X_features,y)

X_test = load_test_data()
X_test_features = model.predict(X_test)
y_test = clasifier.predict(X_test_features)

for i in xrange(0,len(data)):
    data['camera'][i] = back_map[y_test[i]]
    
data.to_csv('ans.csv',index=None)    



