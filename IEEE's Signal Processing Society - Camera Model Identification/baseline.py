import pandas as pd
import numpy as np
import tensorflow as tf
import keras

config = tf.ConfigProto( device_count = {'GPU' : 2, 'CPU':12})
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
from sklearn.cross_validation import train_test_split


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
import keras.backend as K
from keras.callbacks import LearningRateScheduler
from keras.callbacks import Callback
size = 224

from keras.layers.core import Layer
import theano.tensor as T


def get_model():
    model = MobileNet(weights='imagenet',include_top=False,input_shape=(size, size, 3))
    for layer in model.layers:
        layer.trainable = True
    x = model.output
    x = Flatten()(x)
    x.trainable = True
    x = Dense(32,activation='relu')(x)
    x.trainable = True
    x = BatchNormalization(momentum = 0.95)(x)
    x.trainable = True
    x = Dense(64,activation='relu')(x)
    x.trainable = True
    x = BatchNormalization(momentum = 0.95)(x)
    x.trainable = True
    predictions = Dense(10, activation='softmax')(x)
    predictions.trainable = True
    return Model(inputs=model.input, outputs=predictions)


def get_my_model():

    inputs = Input((size,size,3))
    conv1 = Conv2D(32, (4, 4), padding='valid',strides=1,activation='relu')(inputs)
    conv1 = BatchNormalization(axis=3)(conv1)
    pool1 = MaxPool2D(pool_size=(2, 2),strides=2)(conv1)

    conv2 = Conv2D(48, (5, 5), padding='valid', strides=1,activation='relu')(pool1)
    conv2 = BatchNormalization(axis=3)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2),strides=2)(conv2)

    conv3 = Conv2D(64, (5, 5), padding='valid', strides=1,activation='relu')(pool2)
    conv3 = BatchNormalization(axis=3)(conv3)
    pool3 = MaxPool2D(pool_size=(2, 2),strides=2)(conv3)

    conv4 = Conv2D(128, (5, 5), padding='valid', strides=1,activation='relu')(pool3)
    gp4 = GlobalMaxPooling2D()(conv4)

    dense5 = Dense(128,activation='relu')(gp4)
    batch = BatchNormalization()(dense5)

    dense6 = Dense(10,activation='softmax')(batch)
    model = Model(inputs,dense6)
    return model

   

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
            l = int((img.shape[0]/2 - size/2))
            h = int((img.shape[1]/2 - size/2))          
            img_to_apend = img[l:l+size, h:h+size]
                   
                    #np.append((img_to_apend))
            X.append(img_to_apend)
            y.append(map_back[camera])
            
    return X,y

class SGDLearningRateTracker(Callback):
    def on_epoch_end(self, epoch, logs={}):
        optimizer = self.model.optimizer
        lr = K.eval(optimizer.lr * (1. / (1. + optimizer.decay * optimizer.iterations)))
        print('\nLR: {:.6f}\n'.format(lr))
        
def scheduler(epoch):
    if epoch < 10:
        return (.01)
    elif epoch < 20:
        return(.001)
    elif epoch < 50:
        return(.0001)
    else :
        return(.00001)
    return 0.000001        

#X,y = load_train_data() 

#y = to_categorical(y, 10)
#print len(X)
#print len(y)
#print y.shape
#X = np.array(X,np.float32)/255
#X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.1, random_state=247)

vgg19 = get_model()
#print vgg19.summary()
vgg19.load_weights('/home/uberuser/weights/weights_2.hdf5')
#adam = Adam(lr=0.01, decay=0.9)
#change_lr = LearningRateScheduler(scheduler)
#vgg19.compile(loss='categorical_crossentropy', optimizer=adam,metrics=['accuracy'])
#checkpoint_path ='/home/uberuser/weights/weights_2.hdf5'
#checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss', verbose=1, period=1, save_best_only=True)
vgg19.fit(X_train,y_train,nb_epoch=50, batch_size =40, verbose=1,validation_data=(X_test, y_test), callbacks=[checkpoint,change_lr], shuffle=True)
#tmp = vgg19.predict(X_test)
#print y[2]
#print tmp[2]
data = pd.DataFrame(tmp)
data.describe()

ans = path_to_data + 'sample_submission.csv'
ans = pd.read_csv(ans)
ans.head(5)

#model = get_model()
#model.load_weights('/home/uberuser/weights/weights_2.hdf5')
model = vgg19


def load_test(ans):
    X = np.array([],np.float32)
    X = []
    for i in xrange(0,len(ans)):
        img = cv2.imread(path_to_data + 'test/' + ans['fname'][i])
        l = int((img.shape[0]/2 - size/2) )
        h = int((img.shape[1]/2 - size/2)  )          
        img_to_apend = img[l:l+size, h:h+size]
        X.append(img_to_apend)
    return X
test_X = load_test(ans)
test_X = np.array(test_X,np.float32)/255

#print test_X[0]
#fuck = model.predict(X_test)
test = model.predict(test_X)
ans = np.argmax(test,axis=-1)
print ans.shape
sample = path_to_data + 'sample_submission.csv'
sample = pd.read_csv(sample)
for i in xrange(0,2640):
    sample['camera'][i] = back_map[ans[i]]
print sample.head(20)    
sample.to_csv('ans.csv',index=None)
