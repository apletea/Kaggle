import pandas
import cv2
from keras.layers import *
from keras.models import *


data = pandas.read_csv('data_resized_ans.txt',sep=' ')

im_w = 240
im_h = 320

x_train = []
data['x'] = (data['x'] - 120)/120
data['y'] = (data['y'] - 160)/160
y_train = data[data.columns[1:3]].values
print y_train.shape
print y_train

i = 0

for i in xrange(0,len(data)):
    img = cv2.imread('./{}'.format(data['name'][i]),1)
    x_train.append(cv2.resize(img,((im_w,im_h))))

X = np.array(x_train,np.float32)/255
y = np.array(y_train,np.float32)

model = Sequential()
model.add(Convolution2D(1,7,2,input_shape=(320,240,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D((3,3),strides=2))
model.add(Convolution2D(4,(5,5),strides=2,padding='valid'))
model.add(Activation('relu'))
model.add(MaxPooling2D((2,2),strides=2))
model.add(Flatten())
model.add(Dense(2))
model.compile(loss='mean_squared_error', optimizer='sgd')

print model.summary()
pred = model.predict(X)
model.fit(X, y, nb_epoch=100, validation_split=0.1)

model.save('backup.h5')
