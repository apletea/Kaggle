
import tensorflow as tf
import keras

config = tf.ConfigProto( device_count = {'GPU' : 1, 'CPU':16})
sess = tf.Session(config=config)
keras.backend.set_session(sess)


from subprocess import check_output
from keras.applications.vgg19 import VGG19
from keras.models import  Model
from keras.layers import Dense, Dropout, Flatten

import os
from tqdm import tqdm
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import cv2



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
    img = cv2.imread('./train/{}.jpg').format(f)
    label = one_hot_labels[i]
    x_train.append(cv2.resize(img,(im_size,im_size)))
    y_train.append(label)
    i += 1

for f in tqdm(df_test['id'].values):
    img = cv2.imread('../input/test/{}.jpg'.format(f))
    x_test.append(cv2.resize(img, (im_size,im_size)))


y_train_raw = np.array(y_train, np.uint8)
x_train_raw = np.array(x_train,np.float32) / 255
x_test = np.array(x_test, np.float32) / 255

num_class = y_train_raw.shape[1]


X_train, X_valid, Y_train, Y_valid = train_test_split(x_train_raw,y_train_raw,test_size=0.1,random_state=247)


base_line = VGG19(weights='imagenet',include_top=False, input_shape=(im_size,im_size,3))
x = base_line.output
x = Flatten()(x)
predictions = Dense(num_class,activation='softmax')(x)

model = Model(inputs=base_line,outputs=predictions)

for layer in base_line.layers:
    layer.trainable = False

model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])

callbacks_list = [keras.callbacks.EarlyStopping(monitor='val_acc', patience=3, verbose=1)]
print model.summary()

model.fit(X_train,Y_train,epoch=10,validation_data=(X_valid,Y_valid),verbose=1)

preds = model.predict(x_test,verbose=1)

sub = pandas.DataFrame(preds)

col_names = one_hot.columns.values
sub.columns = col_names

sub.insert(0,'id',df_test['id'])
sub.to_csv('final_ans.csv',index=None)
