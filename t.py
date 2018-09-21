import cv2
import numpy as np
import random
#import skimage
#from skimage import data
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


#from shuffle_5 import ShuffleNetV2
from squeezenet import *
from resnet import *
from mobilenetv2 import *


from keras.utils import plot_model

BASE_FOLDER = '/home/dmitry.kamarouski/work/sva_origin/origin_classifier/'
MODEL_NAME = 'sq_4pool_0.90'
input_size = 128
batch_size = 128
epochs = 500
#data_test  = pd.read_csv('~/work/abandoned/indexed/test_data.csv',header=None, sep=' ')
#data_train = pd.read_csv('~/work/abandoned/indexed/true_data.csv',header=None, sep=' ')
data_test  = pd.read_csv('~/work/sva_origin/origin_classifier/test.csv', sep=' ')
data_train = pd.read_csv('~/work/sva_origin/origin_classifier/train_data_2.csv', sep=' ')


import telegram
from keras.callbacks import Callback


class TelegramCallback(Callback):

    def __init__(self):
        super(TelegramCallback, self).__init__()
        self.user_id = 171164240
        self.bot = telegram.Bot('671069810:AAFvY0SDXeoA3hzM_Y-qjLxMCPBh-bEDc6U')
        self.y = []
        self.x = []


    def send_message(self, text):
        try:
            self.bot.send_message(chat_id=self.user_id, text=text)
        except Exception as e:
            print('Message did not send. Error: {}.'.format(e))
    
    def send_graph(self, graphName):
        self.bot.send_photo(chat_id=self.user_id,photo=open(graphName, 'rb'))    
    
    def create_graph_witn_name(self, graphName):
        fig = plt.figure()
        ax = plt.subplot(111)
        ax.plot(self.x, self.y, label='y = val_loss')
        plt.title('Legend inside')
        ax.legend()
        fig.savefig(graphName)

    def on_train_begin(self, logs={}):
        text = 'Start training model {}.'.format(self.model.name)
        self.send_message(text)

    def on_epoch_end(self, epoch, logs={}):
        text = 'Epoch {}.\n'.format(epoch)
        for k, v in logs.items():
            text += '{}: {:.4f}; '.format(k, v)
        self.x.append(epoch)
        self.y.append(logs['val_loss'])
        self.create_graph_witn_name('{}.png'.format(str(epoch)))
        self.send_graph('{}.png'.format(str(epoch)))
        self.send_message(text)


def randomHueSaturationValue(image, hue_shift_limit=(-180, 180),
                             sat_shift_limit=(-255, 255),
                             val_shift_limit=(-255, 255), u=0.5):
    if np.random.random() < u:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(image)
        hue_shift = np.random.uniform(hue_shift_limit[0], hue_shift_limit[1])
        h = cv2.add(h, hue_shift)
        sat_shift = np.random.uniform(sat_shift_limit[0], sat_shift_limit[1])
        s = cv2.add(s, sat_shift)
        val_shift = np.random.uniform(val_shift_limit[0], val_shift_limit[1])
        v = cv2.add(v, val_shift)
        image = cv2.merge((h, s, v))
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    return image

def randomShiftScaleRotate(image, shift_limit=(-0.0625, 0.0625),
                           scale_limit=(-0.1, 0.1),
                           rotate_limit=(-45, 45), aspect_limit=(0, 0),
                           borderMode=cv2.BORDER_CONSTANT, u=0.5):
    if np.random.random() < u:
        height, width, channel = image.shape

        angle = np.random.uniform(rotate_limit[0], rotate_limit[1])  # degree
        scale = np.random.uniform(1 + scale_limit[0], 1 + scale_limit[1])
        aspect = np.random.uniform(1 + aspect_limit[0], 1 + aspect_limit[1])
        sx = scale * aspect / (aspect ** 0.5)
        sy = scale / (aspect ** 0.5)
        dx = round(np.random.uniform(shift_limit[0], shift_limit[1]) * width)
        dy = round(np.random.uniform(shift_limit[0], shift_limit[1]) * height)

        cc = np.math.cos(angle / 180 * np.math.pi) * sx
        ss = np.math.sin(angle / 180 * np.math.pi) * sy
        rotate_matrix = np.array([[cc, -ss], [ss, cc]])

        box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ])
        box1 = box0 - np.array([width / 2, height / 2])
        box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2 + dx, height / 2 + dy])

        box0 = box0.astype(np.float32)
        box1 = box1.astype(np.float32)
        mat = cv2.getPerspectiveTransform(box0, box1)
        image = cv2.warpPerspective(image, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                    borderValue=(0, 0, 0,))

    return image

def randomHorizontalFlip(image, u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 1)
    return image

def calculate_mAP(y_true,y_pred):
    num_classes = y_true.shape[1]
    average_precisions = []
    relevant = K.sum(K.round(K.clip(y_true, 0, 1)))
    tp_whole = K.round(K.clip(y_true * y_pred, 0, 1))
    for index in range(num_classes):
        temp = K.sum(tp_whole[:,:index+1],axis=1)
        average_precisions.append(temp * (1/(index + 1)))
    AP = Add()(average_precisions) / relevant
    mAP = K.mean(AP,axis=0)
    return mAP

def frange(start, stop, step):
    i = start
    while i < stop:
        yield i
        i += step


def ap(y_true, y_pred):
    score = 0
    y_pred = tf.identity(y_pred)
    return calculate_mAP(y_true, y_pred)

def resize(img, desired_size):
    width = img.shape[0]
    height = img.shape[1]
    padd_image = np.zeros((max(width, height),max(width, height),3))
    padd_image[:width,:height] = img
    padd_image = cv2.resize(padd_image, desired_size)
    img = cv2.resize(img,desired_size)
    img[:desired_size[0],:desired_size[1]] = padd_image[:desired_size[0],:desired_size[1]]
    return img

def train_generator():
    while True:
        for start in range(0, len(data_train), batch_size):
            x_batch = []
            y_batch = []
            k = 0
            end = min(start + batch_size, len(data_train))
            ids_train_batch = data_train[start:end]
#            print ids_train_batch[0]
            for x,y in ids_train_batch.iterrows():
     #           print BASE_FOLDER+'{}'.format(y[0])
    
                base_img = cv2.imread(BASE_FOLDER+'{}'.format(y[0]))
                if (base_img is None):
                  continue
                img = cv2.resize(base_img, (input_size, input_size))
                img = randomHueSaturationValue(img,
                                               hue_shift_limit=(-50, 50),
                                               sat_shift_limit=(-5, 5),
                                               val_shift_limit=(-15, 15))
                img = randomShiftScaleRotate(img,
                                                   shift_limit=(-0.0625, 0.0625),
                                                   scale_limit=(-0.1, 0.1),
                                                   rotate_limit=(-0, 0))
                img = randomHorizontalFlip(img)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                zer = np.zeros((input_size, input_size, 1))
                zer[:input_size,:input_size,0] = img
                x_batch.append(zer)
                y_batch.append([y[1],y[2],y[3]])
            if (len(x_batch) != batch_size):
                while (len(x_batch) != batch_size):
                    x_batch.append(x_batch[0])
                    y_batch.append(y_batch[0])
            x_batch = np.array(x_batch, np.float32) / 255
            random_y_train = np.random.rand(batch_size,1)
            random_y_train = np.array(random_y_train)
            y_train_value = y_batch
            y_train_value = np.array(y_train_value)
#            y_batch = to_categorical(y_batch,num_classes=2)
            y_batch = np.array(y_batch)
            yield x_batch, y_batch
            #yield x_batch, y_batch


def valid_generator():
    while True:
        for start in range(0, len(data_test), batch_size):
            x_batch = []
            y_batch = []
            end = min(start + batch_size, len(data_test))
            ids_valid_batch = data_test[start:end]
            for x,y in ids_valid_batch.iterrows():
                base_img = cv2.imread(BASE_FOLDER+'{}'.format(y[0]))
                if (base_img is None):
                    continue
                img = cv2.resize(base_img, (input_size, input_size))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                zer = np.zeros((input_size, input_size, 1))
                zer[:input_size,:input_size,0] = img
                x_batch.append(zer)
                y_batch.append([y[1],y[2],y[3]])
            if (len(x_batch) != batch_size):
                while (len(x_batch) != batch_size):
                    x_batch.append(x_batch[0])
                    y_batch.append(y_batch[0])
            x_batch = np.array(x_batch, np.float32) / 255
            random_y_train = np.random.rand(batch_size,1)
            random_y_train = np.array(random_y_train)
            y_train_value = y_batch
            y_train_value = np.array(y_train_value)
 #           y_batch = to_categorical(y_batch,num_classes=2)
            y_batch = np.array(y_batch)
            yield x_batch, y_batch
            #yield x_batch, y_batch

    
def get_model():
    model  = SqueezeNext50(input_shape=(128,128,1), classes=5)
    x = model.layers[-3].output
    x = Dense(3)(x)
    x = Activation('sigmoid')(x)
    
    model = Model(model.input, x)
    print (model.summary())
    return model

#model = get_model_my()
model = get_model()
#model.load_weights('weights/' + MODEL_NAME)
#model = load_model('constrative-center_loss_or.hdf5', custom_objects={'<lambda>':lambda y_true,y_pred: y_pred, 'fbeta_or' : fbeta_or})
#model = load_model('center_loss_or.hdf5', custom_objects={'fbeta_or':fbeta_or})
#class_weights = [ 0.52155442, 12.09854897]
#model.compile(#loss=weighted_categorical_crossentropy(class_weights),
#              #loss=weighted_binary_crossentropy(class_weights[0],class_weights[1]),
#              #loss=focal_loss_sigmoid,
#              #loss = 'binary_crossentropy',
#              loss = 'categorical_crossentropy',
#              loss_weights=[0.52155442, 12.09854897],
#              optimizer='rmsprop',
#              metrics=['accuracy',fbeta_or]) 
"""
    Center los
"""
model.compile(optimizer=Adam(lr=0.001), loss=["binary_crossentropy"],metrics=['accuracy'])
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
                             filepath='weights/' + MODEL_NAME,
                             save_best_only=True,
                             save_weights_only=True),
            TensorBoard(log_dir='./logs/{}'.format(MODEL_NAME)),
          #  TelegramCallback()]
            ]

model.fit_generator(generator=train_generator(),
                    steps_per_epoch=np.ceil(float(len(data_train)) / float(batch_size)),
                    epochs=epochs,
                    verbose=2,
                    callbacks=callbacks,
                    validation_data=valid_generator(),
                    validation_steps=np.ceil(float(len(data_test)) / float(batch_size)),
                    #class_weight = class_weights
                    #class_weight=class_weights
                   )
model.save('models/{}.hdf5'.format(MODEL_NAME))
