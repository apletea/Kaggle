import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pylab
from keras.models import Sequential
from keras.layers import Dense,Dropout,Lambda,Flatten
from keras.optimizers import Adam,RMSprop
from sklearn.model_selection import train_test_split
from keras.models import  Sequential
from keras.layers.core import  Lambda , Dense, Flatten, Dropout
from keras.callbacks import EarlyStopping
from keras.layers import BatchNormalization, Convolution2D , MaxPooling2D
from subprocess import check_output
from keras.preprocessing import image
test = pd.read_csv('test.csv')
train = pd.read_csv('train.csv')

X_train = (train.ix[:,1:].values).astype('float32')
Y_train = train.ix[:,0].values.astype('int32')
X_test = test.values.astype('float32')

X_train = X_train.reshape(X_train.shape[0],28,28)

for i in range(6, 9):
    plt.subplot(330 + (i+1))
    plt.imshow(X_train[i], cmap=plt.get_cmap('gray'))
    plt.title(Y_train[i])

X_train = X_train.reshape(X_train.shape[0], 28, 28,1)

X_test = X_test.reshape(X_test.shape[0], 28, 28,1)

mean_px = X_train.mean().astype(np.float32)
std_px = X_train.std().astype(np.float32)

def standardize(x):
    return (x-mean_px)/std_px




from keras.utils.np_utils import to_categorical
y_train= to_categorical(Y_train)
num_classes = y_train.shape[1]

plt.title(y_train[9])
plt.plot(y_train[9])
plt.xticks(range(10))

seed = 43
np.random.seed(seed)

pylab.show()

model=Sequential()
model.add(Lambda(standardize,input_shape=(28,28,1)))
model.add(Flatten())
model.add(Dense(10,activation='softmax'))
model.compile(optimizer=RMSprop(lr=0.001),loss='categorical_crossentropy',metrics=['accuracy'])

gen = image.ImageDataGenerator()

