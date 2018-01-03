import pandas
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



df_train =pandas.read_csv('labels.csv')
df_test = pandas.read_csv('sample_submission.csv')


target_series = pandas.Series(df_train['breed'])
one_hot = pandas.get_dummies(target_series, sparse=True)

one_hot_labels = np.asarray(one_hot)

im_size = 224

x_train = []
y_train = []
x_test = []

i = 0

for f, breed in tqdm(df_train.values):
    img = cv2.imread(('./train/{}.jpg').format(f))
    label = one_hot_labels[i]
    x_train.append(cv2.resize(img,(im_size,im_size)))
    y_train.append(label)
    i += 1

for f in tqdm(df_test['id'].values):
    img = cv2.imread('./test/{}.jpg'.format(f))
    x_test.append(cv2.resize(img, (im_size,im_size)))



y_train_raw = np.array(y_train, np.uint8)
x_train_raw = np.array(x_train,np.float32) / 255
x_test = np.array(x_test, np.float32) / 255

num_class = y_train_raw.shape[1]



def get_features(model, data):
    model = model(include_top=False, input_shape=(im_size, im_size, 3), weights='imagenet')

    inputs = Input((im_size,im_size,3))

    x = inputs
    x = Lambda(preprocess_input, name='preprocessing')(x)
    x = model(x)
    x = GlobalAveragePooling2D()(x)
    model = Model(inputs,x)

    features = model.predict(data,batch_size=64,verbose=1)
    return features


vgg_features = get_features(VGG19,data=x_train_raw)
inception_features = get_features(InceptionV3,data=x_train_raw)
xception_features = get_features(Xception,data=x_train_raw)
resnet_features = get_features(ResNet50, data=x_train_raw)

features = np.concatenate([vgg_features,inception_features,xception_features,resnet_features], axis=-1)

inputs = Input(features.shape[1:])
x = inputs
x = Dropout(0.5)(x)
x = Dense(120, activation='softmax')(x)
model = Model(inputs,x)
model.compile(optimizer='adam',
              loss = 'categorical_crossentropy',
              metrics=['accuracy'])
h = model.fit(features,y_train_raw,batch_size=128,epoch=10,validation_split=0.1)

model.save('drapun.h5')


# preds = model.predict(x_test,verbose=1)
# 
# 
# sub = pandas.DataFrame(preds)
# 
# col_names = one_hot.columns.values
# sub.columns = col_names
# 
# sub.insert(0,'id',df_test['id'])
# sub.to_csv('final_ans.csv',index=None)



