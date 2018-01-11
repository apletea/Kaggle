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
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import StratifiedKFold


def augment_brightness_camera_images(image):
    image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    random_bright = .25+np.random.uniform()
    #print(random_bright)
    image1[:,:,2] = image1[:,:,2]*random_bright
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    return image1

def transform_image(img,ang_range,shear_range,trans_range,brightness=0):
    '''
    This function transforms images to generate new images.
    The function takes in following arguments,
    1- Image
    2- ang_range: Range of angles for rotation
    3- shear_range: Range of values to apply affine transform to
    4- trans_range: Range of values to apply translations over.

    A Random uniform distribution is used to generate different parameters for transformation

    '''
    # Rotation

    ang_rot = np.random.uniform(ang_range)-ang_range/2
    rows,cols,ch = img.shape
    Rot_M = cv2.getRotationMatrix2D((cols/2,rows/2),ang_rot,1)

    # Translation
    tr_x = trans_range*np.random.uniform()-trans_range/2
    tr_y = trans_range*np.random.uniform()-trans_range/2
    Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])

    # Shear
    pts1 = np.float32([[5,5],[20,5],[5,20]])

    pt1 = 5+shear_range*np.random.uniform()-shear_range/2
    pt2 = 20+shear_range*np.random.uniform()-shear_range/2

    # Brightness


    pts2 = np.float32([[pt1,5],[pt2,pt1],[5,pt2]])

    shear_M = cv2.getAffineTransform(pts1,pts2)

    img = cv2.warpAffine(img,Rot_M,(cols,rows))
    img = cv2.warpAffine(img,Trans_M,(cols,rows))
    img = cv2.warpAffine(img,shear_M,(cols,rows))

    if brightness == 1:
      img = augment_brightness_camera_images(img)

    return img

def augment(img):
    ans = []
    for i in xrange(0,200):
        imgc = img.copy()
        imgc = transform_image(imgc,20,10,5,brightness=1)
        ans.append(imgc)
    return ans

# def MultiLogLoss(y_pred, y_true):

def multiclass_loss(y_true, y_pred):
    '''
    Our custom multiclass loss function
    '''
    EPS = 1e-5
    y_pred = K.clip(y_pred, EPS, 1 - EPS)
    return -K.mean((1 - y_true) * K.log(1 - y_pred) + y_true * K.log(y_pred))

df_train =pandas.read_csv('labels.csv')


df_test = pandas.read_csv('sample_submission.csv')
target_series = pandas.Series(df_train['breed'])

one_hot = pandas.get_dummies(target_series, sparse=True)

one_hot_labels = np.asarray(one_hot)

im_size = 250
x_train = []
y_train = []

x_test = []

i = 0


for f, breed in tqdm(df_train.values):
    img = cv2.imread(('./train/{}.jpg').format(f))
    label = one_hot_labels[i]
    x_train.append(cv2.resize(img, (im_size, im_size)))
    y_train.append(label)
    i += 1

for f in tqdm(df_test['id'].values):
    img = cv2.imread('./test/{}.jpg'.format(f))
    x_test.append(cv2.resize(img, (im_size,im_size)))



y_train_raw = np.array(y_train, np.uint8)
x_train_raw = np.array(x_train,np.float32) / 255
x_test = np.array(x_test, np.float32) / 255

num_class = y_train_raw.shape[1]



def get_features(MODEL, data):
    model = MODEL(include_top=False, input_shape=(im_size, im_size, 3), weights='imagenet')

    inputs = Input((im_size,im_size,3))

    x = inputs
    x = Lambda(preprocess_input, name='preprocessing')(x)
    x = model(x)
    x = GlobalAveragePooling2D()(x)
    model = Model(inputs,x)

    features = model.predict(data,batch_size=64,verbose=1)
    print features.shape
    return features


# vgg_features = get_features(VGG19,data=x_train_raw)
# inception_features = get_features(InceptionV3,data=x_train_raw)
# xception_features = get_features(Xception,data=x_train_raw)
# resnet_features = get_features(ResNet50, data=x_test)
#
#
# features = np.concatenate([inception_features,xception_features], axis=-1)

# inputs = Input(features.shape[1:])
# x = inputs
# x = Dropout(0.5)(x)
# x = Dense(120, activation='softmax')(x)
# model = Model(inputs,x)
# model.compile(optimizer='adam',
#               loss = 'categorical_crossentropy',
#               metrics=['accuracy'])
# h = model.fit(features,y_train_raw,batch_size=128,nb_epoch=10,validation_split=0.1)



skf = StratifiedKFold(n_splits=10,shuffle=True)
cvscores = []
for index, (train_indices, val_indices) in enumerate(skf.split(x_train_raw,y_train_raw)):
    print "Training on fold " + str(index+1) + "/10..."

    xtrain, xval = x_train_raw[train_indices],x_train_raw[val_indices]
    ytrain, yval = y_train_raw[train_indices],y_train_raw[val_indices]

    base = VGG19(include_top=False, weights='imagenet', input_shape=(im_size, im_size, 3))
    x = base.output
    x = Dropout(0.5)(x)
    x = Flatten()(x)
    x = Dense(1000, activation='selu')(x)
    x = Dropout(0.5)(x)
    x = Dense(num_class, activation='softmax')(x)
    model = Model(inputs=base.inputs,outputs=x)

    for layer in base.layers:
        layer.trainable = False

    model.compile(loss=multiclass_loss, optimizer='adam')
    checkpointer = ModelCheckpoint(filepath='./tmp/weights_{}.hdf5'.format(index), verbose=1, save_best_only=True)

    model.fit(x_train_raw,y_train_raw,epochs=30,callbacks=[checkpointer])

    scores = model.evaluate(xval,yval)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
    cvscores.append(scores[1] * 100)
    
print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))



model.save('orig.h5')

model.load_weights('./tmp/weights.hdf5')
# model = load_model('orig.h5', custom_objects={"multiclass_loss":multiclass_loss})


preds = model.predict(x_test,verbose=1)


sub = pandas.DataFrame(preds)

col_names = one_hot.columns.values
sub.columns = col_names

sub.insert(0,'id',df_test['id'])
sub.to_csv('final_ans.csv',index=None)



