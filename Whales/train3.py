import cv2
import numpy as np
import random
import skimage
from skimage import data
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib as plt
plt.use('Agg')
from matplotlib import pyplot as plt
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
from sklearn.metrics import roc_curve, auc, average_precision_score, precision_recall_curve
from sklearn.utils import class_weight
from keras.models import *
from keras.optimizers import *
from keras.applications import *
import random
import argparse
import numpy as np
import gc
from keras.initializers import glorot_uniform
import telegram
from keras.callbacks import Callback

print ('train3.py -base_folder $path_to_train')

parser = argparse.ArgumentParser(description='set input arguments')
parser.add_argument('-base_folder', action='store', dest='base_folder',type=str, default='./train/')
parser.add_argument('-input_size',  action='store', dest='input_size', type=int, default=384)
parser.add_argument('-batch_size',  action='store', dest='batch_size', type=int, default=18)
parser.add_argument('-epochs',      action='store', dest='epochs',     type=int, default=15)
parser.add_argument('-model_name',  action='store', dest='model_name', type=str, default='Xception_whales_iteration_0')
parser.add_argument('-train_csv',   action='store', dest='train_csv',  type=str, default='../../train.csv')
parser.add_argument('-name_id',     action='store', dest='name_id',    type=str, default='name_id.csv')
args = parser.parse_args()

BASE_FOLDER = args.base_folder
MODEL_NAME  = args.model_name
input_size  = args.input_size
batch_size  = args.batch_size
epochs      = args.epochs

data = pd.read_csv(args.train_csv)

cl  = pd.read_csv(args.name_id, sep=' ')
name_ids = dict([(i,b) for i,b in zip(cl.name, cl.id)])
num_classes = 5005

st = set(pd.read_csv('classes.csv', header=None)[0])

if (not os.path.exists('val_train.csv')):
    data_test  = pd.DataFrame(columns = data.columns)
    data_train = pd.DataFrame(columns = data.columns)
    for i in range(len(data)):
       if (data.iloc[i].Id in st):
           if (random.random() > 0.95):
               data_test = data_test.append(data.iloc[i])
           else:
               data_train = data_train.append(data.iloc[i])
       else:
           data_train = data_train.append(data.iloc[i])
             
    data_test.to_csv('val_train.csv', index=None)
    data_train.to_csv('t_train.csv', index=None)
  
data_test  = pd.read_csv('val_train.csv')
data_train = pd.read_csv('t_train.csv')

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



class TelegramCallback(Callback):

    def __init__(self):
        super(TelegramCallback, self).__init__()
        self.user_id = 171164240
        self.bot = telegram.Bot('671069810:AAFvY0SDXeoA3hzM_Y-qjLxMCPBh-bEDc6U')
        self.y = []
        self.x = []
        self.pred_y = [] 


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


def resize(img, desired_size):
    width = img.shape[0]
    height = img.shape[1]
    padd_image = np.zeros((max(width, height),max(width, height),3))
    padd_image[:width,:height] = img
    padd_image = cv2.resize(padd_image, desired_size)
    img = cv2.resize(img,desired_size)
    img[:desired_size[0],:desired_size[1]] = padd_image[:desired_size[0],:desired_size[1]]
    return img

def train_generator(num_classes, name_ids):
    while True:
        for start in range(0, len(data_train), batch_size):
            x_batch = []
            y_batch = []
            k = 0
            end = min(start + batch_size, len(data_train))
            ids_train_batch = data_train[start:end]
#            print ids_train_batch[0]
            for x,y in ids_train_batch.iterrows():
                base_img = cv2.imread(os.path.join(BASE_FOLDER, y[0]), 0)
                if (base_img is None):
                    print(BASE_FOLDER+'{}'.format(y[0]))
                    print('None')
                    continue
                img = cv2.resize(base_img, (input_size, input_size))
#               img = randomHueSaturationValue(img,
#                                              hue_shift_limit=(-50, 50),
#                                              sat_shift_limit=(-5, 5),
#                                              val_shift_limit=(-15, 15))
#               img = randomShiftScaleRotate(img,
#                                                  shift_limit=(-0.0625, 0.0625),
#                                                  scale_limit=(-0.1, 0.1),
#                                                  rotate_limit=(-0, 0))
#               img = randomHorizontalFlip(img)
                zeros = np.zeros((input_size, input_size, 1))
                zeros[:input_size,:input_size,0] = img
                img = zeros
                x_batch.append(img)
                y_batch.append(name_ids[y[1]])
            if (len(x_batch) != batch_size):
                while (len(x_batch) != batch_size):
                    x_batch.append(x_batch[0])
                    y_batch.append(y_batch[0])
            x_batch = np.array(x_batch, np.float32) / 255
            random_y_train = np.random.rand(batch_size,1)
            random_y_train = np.array(random_y_train)
            y_train_value = y_batch
            is_whale = np.array(1 - (np.array(y_batch)==0), np.int8)
            y_train_value = np.array(y_train_value)
            not_y_train_value = [[i for i in range(num_classes) if i!=b] for b in y_train_value ]
            not_y_train_value = np.array([np.array(b) for b in not_y_train_value])
            y_batch = to_categorical(y_batch,num_classes=num_classes)
            yield [x_batch] + [y_train_value] , [y_batch, random_y_train]



def valid_generator(num_classes, name_ids):
    while True:
        for start in range(0, len(data_test), batch_size):
            x_batch = []
            y_batch = []
            end = min(start + batch_size, len(data_test))
            ids_valid_batch = data_test[start:end]
            for x,y in ids_valid_batch.iterrows():
                base_img = cv2.imread(os.path.join(BASE_FOLDER, y[0]),0)
                if (base_img is None):
                    continue
                img = cv2.resize(base_img, (input_size, input_size))
                zeros = np.zeros((input_size, input_size, 1))
                zeros[:input_size,:input_size,0] = img
                img = zeros
                x_batch.append(img)
                y_batch.append(name_ids[y[1]])
            if (len(x_batch) != batch_size):
                while (len(x_batch) != batch_size):
                    x_batch.append(x_batch[0])
                    y_batch.append(y_batch[0])
            x_batch = np.array(x_batch, np.float32) / 255
            random_y_train = np.random.rand(batch_size,1)
            random_y_train = np.array(random_y_train)
            y_train_value = y_batch
            is_whale = np.array(1- (np.array(y_batch)==0), np.int8)
            y_train_value = np.array(y_train_value)
            not_y_train_value = [[i for i in range(num_classes) if i!=b] for b in y_train_value ]
            not_y_train_value = np.array([np.array(b) for b in not_y_train_value])
            y_batch = to_categorical(y_batch,num_classes=num_classes)
            yield [x_batch] + [y_train_value] , [y_batch, random_y_train]
            #yield x_batch, y_batch

 


def get_model_best_cl(num_classes):
    vec_dim = 128
    #model = Xception(input_shape=(input_size,input_size,3), weights='imagenet', include_top=False)
    model = load_model('mpiotte-standard.model')
    model = model.get_layer('model_1')
    model = Model(model.get_input_at(1), model.get_output_at(1))
    ip1 = Dense(vec_dim)(model.output)
    ip1 = PReLU()(ip1)
    ip2 = Dense(num_classes)(ip1) 
    ip2 = Activation('softmax')(ip2)
    emb_inputs = []
    for i in range(1):
        emb_inputs.append(Input(shape=(1,), name='input_{}'.format(str(i+6))))

#   centers = load_model('models/whale_emb.h5') 
    centers = Embedding(num_classes, vec_dim)
    centers_arr = []
    for i in range(1):
        centers_arr.append(centers(emb_inputs[i]))
    l2_loss = Lambda(lambda x:K.mean(K.sum(K.square(x[0]-x[1][:,0]),keepdims=True,axis=(1)),axis=0, keepdims=True),name='l2_loss')([ip1] + centers_arr)
    model_centerloss = Model(inputs=[model.input] + emb_inputs , outputs=[ip2,l2_loss]) 
    return model_centerloss

model = get_model_best_cl(num_classes)
model.compile(optimizer=SGD(lr=0.1), loss=["categorical_crossentropy", lambda y_true,y_pred: y_pred],loss_weights=[1,0.0004],metrics=['accuracy'])
print(model.summary())


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
                             filepath=MODEL_NAME,
                             save_best_only=True,
                             save_weights_only=True),
            TensorBoard(log_dir='./logs/{}'.format(MODEL_NAME)),
            TelegramCallback()
            ]

model.fit_generator(generator=train_generator(num_classes, name_ids),
                   steps_per_epoch=np.ceil(float(len(data_train)) / float(batch_size)),
                   epochs=1,
                   verbose=1,
                   callbacks=callbacks,
                   validation_data=valid_generator(num_classes, name_ids),
                   validation_steps=np.ceil(float(len(data_test)) / float(batch_size)),
                  )

model.compile(optimizer=SGD(lr=0.1), loss=["categorical_crossentropy", lambda y_true,y_pred: y_pred],loss_weights=[1,0.4],metrics=['accuracy'])
model.fit_generator(generator=train_generator(num_classes, name_ids),
                   steps_per_epoch=np.ceil(float(len(data_train)) / float(batch_size)),
                   epochs=epochs,
                   verbose=1,
                   callbacks=callbacks,
                   validation_data=valid_generator(num_classes, name_ids),
                   validation_steps=np.ceil(float(len(data_test)) / float(batch_size)),
                  )


in_model = Model(model.input[0], model.output[0])
in_model.save('inferece_{}.hdf5'.format(MODEL_NAME))

