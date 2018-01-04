import pandas
import cv2
from keras.layers import *
from keras.models import *
from keras.layers import *
from keras.models import *
from keras.applications import *
from keras.optimizers import *
from keras.regularizers import *

#
# def square(y_true, y_pred):
#     return (y_true[0]-y_pred[0])**2 + (y_true[1]-y_pred[1])**2
#
#
# data = pandas.read_csv('data_resized_ans.txt',sep=' ')
#
# im_w = 240
# im_h = 320
#
# x_train = []
# data['x'] = (data['x'] - 120)/120
# data['y'] = (data['y'] - 160)/160
# y_train = data[data.columns[1:3]].values
# print y_train.shape
# print y_train
#
# i = 0
#
# for i in xrange(0,len(data)):
#     img = cv2.imread('./{}'.format(data['name'][i]),3)
#     x_train.append(cv2.resize(img,((im_w,im_h))))
#
# X = np.array(x_train,np.float32)/255
# y = np.array(y_train,np.float32)
#
# base = VGG16(include_top=False,weights='imagenet',input_shape=(320,240,3))
# x = base.output
# x = Flatten()(x)
# predictions = Dense(2)(x)
#
# model = Model(inputs=base.inputs,outputs=predictions)
# for layer in base.layers:
#     layer.trainable = False
# model.compile(loss='mean_squared_error', optimizer='adam')
#
# print model.summary()
# pred = model.predict(X)
# model.fit(X, y, nb_epoch=9, validation_split=0.1)
#
# model.save('t2.h5')

from keras.applications.inception_v3 import preprocess_input

# model = load_model('t2.h5')
# vc = cv2.VideoCapture('/home/please/VID_20180103_172834.mp4')
# while(True):
#     _,img = vc.read()
#     X = []
#     img = cv2.resize(img,(240,320))
#
#     rows, cols = 320,240
#     matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), 90, 1)
#     img = cv2.warpAffine(img, matrix, (cols, rows))
#
#     X.append(cv2.resize(img,((240,320))))
#     X = np.array(X,np.float32)/255
#     pred = model.predict(X)
#     x = pred[0][0]*120+120
#     y = pred[0][1]*160+160
#     cv2.circle(img,(int(x),int(y)),10,(0,0,255))
#     cv2.imshow('res',img)
#     cv2.waitKey(25)
import pandas as pd
from keras.optimizers import RMSprop
from sklearn.utils import shuffle

FTRAIN = 'training.csv'
FTEST  = 'test.csv'
def load(test=False, cols=None):

    fname = FTEST if test else FTRAIN
    df = pd.read_csv(fname)
    df['Image'] = df['Image'].apply(lambda  im: np.fromstring(im, sep=' '))

    if cols:
        df = df[list(cols)+['Image']]

    # print df.count()
    # df = df.dropna()
    print df
    X = np.array([np.array(band).astype(np.float32).reshape(96,96) for band in df['Image']])
    X = np.concatenate([X[:, :, :, np.newaxis],X[:, :, :, np.newaxis],X[:, :, :, np.newaxis]],axis=-1)
    # X = np.vstack(df['Image'].values) / 255
    # X = X.astype(np.float32)
    if not test:
        y = df[df.columns[:-1]].values
        y = (y-48)/48
        X,y = shuffle( X, y, random_state=42)
        y = y.astype(np.float32)
    else :
        y = None

    return X,y


def laod2d(test=False, cols=None):

    X,y = load(test,cols)
    # X = X.reshape(-1,1,96,96)
    X = X/255
    return X,y

# X,y = laod2d(test=False)
# print y[0]
# base_model = VGG19(include_top=False,weights='imagenet',input_shape=(96,96,3))
# x = base_model.output
# x = Dropout(0,5)(x)
# x = Flatten()(x)
# x = Dense(100,activation='selu')(x)
# x = Dropout(0,5)(x)
# x = Dense(100,activation='selu')(x)
# x = Dropout(0,5)(x)
# x = Dense(30)(x)
#
# model = Model(inputs=base_model.inputs, outputs=x)
#
# for layer in base_model.layers :
#     layer.trainable = False
#
#
#
#
# rms = RMSprop(lr=0.001, rho=0.9, decay=0.0)
# model.compile(loss='mean_squared_error', optimizer=rms)
#
# pred = model.predict(X)
# print pred[0]
#
# print model.summary()
# model.fit(X,y, nb_epoch=100,validation_split=0.1)
# model.save('model.h5')


model = load_model('model.h5')
X_test, _ = laod2d(test=True)
y_test = model.predict(X_test)
print X_test
map = ['left_eye_center_x','left_eye_center_y','right_eye_center_x','right_eye_center_y','left_eye_inner_corner_x','left_eye_inner_corner_y','left_eye_outer_corner_x','left_eye_outer_corner_y','right_eye_inner_corner_x','right_eye_inner_corner_y','right_eye_outer_corner_x','right_eye_outer_corner_y','left_eyebrow_inner_end_x','left_eyebrow_inner_end_y','left_eyebrow_outer_end_x','left_eyebrow_outer_end_y','right_eyebrow_inner_end_x','right_eyebrow_inner_end_y','right_eyebrow_outer_end_x','right_eyebrow_outer_end_y','nose_tip_x','nose_tip_y','mouth_left_corner_x','mouth_left_corner_y','mouth_right_corner_x','mouth_right_corner_y','mouth_center_top_lip_x','mouth_center_top_lip_y','mouth_center_bottom_lip_x','mouth_center_bottom_lip_y']
res = pd.read_csv('IdLookupTable.csv')
map  ={map[i]:i for i in xrange(0,30)}
print map
print res
for i in xrange(0,len(res)):
    print res['RowId'][i]
    res['Location'][i] = y_test[res['ImageId'][i]-1][map[res['FeatureName'][i]]]

#
ans = res[['RowId'],['Location']]
ans.to_csv('ans.csv',index=None)
