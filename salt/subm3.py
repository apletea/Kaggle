import cv2
import numpy as np
import random
import skimage
from skimage import data
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

from keras.callbacks import ModelCheckpoint , EarlyStopping, ReduceLROnPlateau, TensorBoard
from keras.utils.np_utils import to_categorical
from keras import backend as K
from keras.models import load_model
#from sklearn.metrics import roc_curve, auc, average_precision_score, precision_recall_curve
#from sklearn.utils import class_weight
from keras.models import *
from keras.optimizers import *
from keras.applications import *
import telegram
from keras.callbacks import Callback


version = 5
basic_name = 'Unet_resnet_v{}'.format(str(version))
save_model_name = basic_name + '.h5'
submission_file = basic_name + '.csv'


img_size_ori = 101
img_size_target = 101
net_size = 128
def upsample(img):# not used
    reflect = cv2.copyMakeBorder(img,13,14,13,14,cv2.BORDER_REFLECT)
    return reflect
    
def downsample(img):# not used
    ans = img[13:114,13:114]
    return ans


def unpad_image(images):
    ans = np.zeros((101,101,3))
    ans = images[:101,:101]
    return ans

def padd_image(image):
    channels = 3
    ans = np.zeros((128,128,channels))
    ans[:101,:101] = image
    return ans

train_df = pd.read_csv('/home/dmitry.kamarouski/work/rd/salt/train/train.csv', header=None)
val_df  = pd.read_csv('/home/dmitry.kamarouski/work/rd/salt/train/test.csv', header=None)
test_df = pd.read_csv('/home/dmitry.kamarouski/work/rd/salt/test/test.csv',header=None)
#train_df["images"] = [np.array(load_img("train/images/{}.png".format(idx), grayscale=True))   / 255 for idx in train_df.index]
#train_df["masks"]  = [np.array(load_img("train/masks/{}.png".format(idx) ,  grayscale=True))  / 255 for idx in train_df.index]
#val_df["images"]   = [np.array(load_img("train/images/{}.png".format(idx), grayscale=True))   / 255 for idx in val_df.index  ]
#val_df["masks"]    = [np.array(load_img("train/masks/{}.png".format(idx) ,  grayscale=True))  / 255 for idx in val_df.index  ]

def BatchActivate(x):
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def convolution_block(x, filters, size, strides=(1,1), padding='same', activation=True):
    x = Conv2D(filters, size, strides=strides, padding=padding)(x)
    if activation == True:
        x = BatchActivate(x)
    return x

def residual_block(blockInput, num_filters=16, batch_activate = False):
    x = BatchActivate(blockInput)
    x = convolution_block(x, num_filters, (3,3) )
    x = convolution_block(x, num_filters, (3,3), activation=False)
    x = Add()([x, blockInput])
    if batch_activate:
        x = BatchActivate(x)
    return x

def build_model(input_layer, start_neurons, DropoutRatio = 0.5):
    # 128 -> 64
    conv1 = Conv2D(start_neurons * 1, (3, 3), activation=None, padding="same")(input_layer)
    conv1 = residual_block(conv1,start_neurons * 1, True)
    conv1 = residual_block(conv1,start_neurons * 1, True)
    conv1 = residual_block(conv1,start_neurons * 1, True)
    pool1 = MaxPooling2D((2, 2))(conv1)
    pool1 = Dropout(DropoutRatio/2)(pool1)

    # 64 -> 32
    conv2 = Conv2D(start_neurons * 2, (3, 3), activation=None, padding="same")(pool1)
    conv2 = residual_block(conv2,start_neurons * 2, True)
    conv2 = residual_block(conv2,start_neurons * 2, True)
    conv2 = residual_block(conv2,start_neurons * 2, True)
    pool2 = MaxPooling2D((2, 2))(conv2)
    pool2 = Dropout(DropoutRatio)(pool2)

    # 32 -> 16
    conv3 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same")(pool2)
    conv3 = residual_block(conv3,start_neurons * 4, True)
    conv3 = residual_block(conv3,start_neurons * 4, True)
    conv3 = residual_block(conv3,start_neurons * 4, True)
    pool3 = MaxPooling2D((2, 2))(conv3)
    pool3 = Dropout(DropoutRatio)(pool3)

    # 16 -> 8
    conv4 = Conv2D(start_neurons * 8, (3, 3), activation=None, padding="same")(pool3)
    conv4 = residual_block(conv4,start_neurons * 8, True)
    conv4 = residual_block(conv4,start_neurons * 8, True)
    conv4 = residual_block(conv4,start_neurons * 8, True)
    pool4 = MaxPooling2D((2, 2))(conv4)
    pool4 = Dropout(DropoutRatio)(pool4)

    # Middle
    convm = Conv2D(start_neurons * 16, (3, 3), activation=None, padding="same")(pool4)
    convm = residual_block(convm,start_neurons * 16, True)
    convm = residual_block(convm,start_neurons * 16, True)
    convm = residual_block(convm,start_neurons * 16, True)
    
    # 8 -> 16
    deconv4 = Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding="same")(convm)
    uconv4 = concatenate([deconv4, conv4])
    uconv4 = Dropout(DropoutRatio)(uconv4)
    
    uconv4 = Conv2D(start_neurons * 8, (3, 3), activation=None, padding="same")(uconv4)
    uconv4 = residual_block(uconv4,start_neurons * 8)
    uconv4 = residual_block(uconv4,start_neurons * 8, True)
    
    # 16 -> 32
    #deconv3 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(uconv4)
    deconv3 = Conv2DTranspose(start_neurons * 4, (2, 2), strides=(2, 2), padding="valid")(uconv4)
    uconv3 = concatenate([deconv3, conv3])    
    uconv3 = Dropout(DropoutRatio)(uconv3)
    
    uconv3 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same")(uconv3)
    uconv3 = residual_block(uconv3,start_neurons * 4)
    uconv3 = residual_block(uconv3,start_neurons * 4, True)

    # 32 -> 64
    deconv2 = Conv2DTranspose(start_neurons * 2, (2, 2), strides=(2, 2), padding="same")(uconv3)
    uconv2 = concatenate([deconv2, conv2])
        
    uconv2 = Dropout(DropoutRatio)(uconv2)
    uconv2 = Conv2D(start_neurons * 2, (3, 3), activation=None, padding="same")(uconv2)
    uconv2 = residual_block(uconv2,start_neurons * 2)
    uconv2 = residual_block(uconv2,start_neurons * 2, True)
    
    # 64 -> 128
    #deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same")(uconv2)
    deconv1 = Conv2DTranspose(start_neurons * 1, (2, 2), strides=(2, 2), padding="valid")(uconv2)
    uconv1 = concatenate([deconv1, conv1])
    
    uconv1 = Dropout(DropoutRatio)(uconv1)
    uconv1 = Conv2D(start_neurons * 1, (3, 3), activation=None, padding="same")(uconv1)
    uconv1 = residual_block(uconv1,start_neurons * 1)
    uconv1 = residual_block(uconv1,start_neurons * 1, True)
    
    #uconv1 = Dropout(DropoutRatio/2)(uconv1)
    #output_layer = Conv2D(1, (1,1), padding="same", activation="sigmoid")(uconv1)
    output_layer_noActi = Conv2D(1, (1,1), padding="same", activation=None)(uconv1)
    output_layer =  Activation('sigmoid')(output_layer_noActi)
    
    return output_layer

class SGDRScheduler(Callback):
    '''Cosine annealing learning rate scheduler with periodic restarts.
    # Usage
        ```python
            schedule = SGDRScheduler(min_lr=1e-5,
                                     max_lr=1e-2,
                                     steps_per_epoch=np.ceil(epoch_size/batch_size),
                                     lr_decay=0.9,
                                     cycle_length=5,
                                     mult_factor=1.5)
            model.fit(X_train, Y_train, epochs=100, callbacks=[schedule])
        ```
    # Arguments
        min_lr: The lower bound of the learning rate range for the experiment.
        max_lr: The upper bound of the learning rate range for the experiment.
        steps_per_epoch: Number of mini-batches in the dataset. Calculated as `np.ceil(epoch_size/batch_size)`. 
        lr_decay: Reduce the max_lr after the completion of each cycle.
                  Ex. To reduce the max_lr by 20% after each cycle, set this value to 0.8.
        cycle_length: Initial number of epochs in a cycle.
        mult_factor: Scale epochs_to_restart after each full cycle completion.
    # References
        Blog post: jeremyjordan.me/nn-learning-rate
        Original paper: http://arxiv.org/abs/1608.03983
    '''
    def __init__(self,
                 min_lr,
                 max_lr,
                 steps_per_epoch,
                 lr_decay=1,
                 cycle_length=10,
                 mult_factor=2):

        self.min_lr = min_lr
        self.max_lr = max_lr
        self.lr_decay = lr_decay

        self.batch_since_restart = 0
        self.next_restart = cycle_length

        self.steps_per_epoch = steps_per_epoch

        self.cycle_length = cycle_length
        self.mult_factor = mult_factor

        self.history = {}

    def clr(self):
        '''Calculate the learning rate.'''
        fraction_to_restart = self.batch_since_restart / (self.steps_per_epoch * self.cycle_length)
        lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + np.cos(fraction_to_restart * np.pi))
        return lr

    def on_train_begin(self, logs={}):
        '''Initialize the learning rate to the minimum value at the start of training.'''
        logs = logs or {}
        K.set_value(self.model.optimizer.lr, self.max_lr)

    def on_batch_end(self, batch, logs={}):
        '''Record previous batch statistics and update the learning rate.'''
        logs = logs or {}
        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        self.batch_since_restart += 1
        K.set_value(self.model.optimizer.lr, self.clr())

    def on_epoch_end(self, epoch, logs={}):
        '''Check for end of current cycle, apply restarts when necessary.'''
        if epoch + 1 == self.next_restart:
            self.batch_since_restart = 0
            self.cycle_length = np.ceil(self.cycle_length * self.mult_factor)
            self.next_restart += self.cycle_length
            self.max_lr *= self.lr_decay
            self.best_weights = self.model.get_weights()

    def on_train_end(self, logs={}):
        '''Set weights to the values from the end of the most recent cycle for best performance.'''
        self.model.set_weights(self.best_weights)

from keras import backend as K
from keras.callbacks import TensorBoard

class LRTensorBoard(TensorBoard):
    def __init__(self, log_dir):  # add other arguments to __init__ if you need
        super(LRTensorBoard, self).__init__(log_dir=log_dir)

    def on_epoch_end(self, epoch, logs=None):
        logs.update({'lr': K.eval(self.model.optimizer.lr)})
        super(LRTensorBoard, self).on_epoch_end(epoch, logs)



def get_iou_vector(A, B):
    batch_size = A.shape[0]
    metric = []
    for batch in range(batch_size):
        t, p = A[batch]>0, B[batch]>0
#         if np.count_nonzero(t) == 0 and np.count_nonzero(p) > 0:
#             metric.append(0)
#             continue
#         if np.count_nonzero(t) >= 1 and np.count_nonzero(p) == 0:
#             metric.append(0)
#             continue
#         if np.count_nonzero(t) == 0 and np.count_nonzero(p) == 0:
#             metric.append(1)
#             continue
        
        intersection = np.logical_and(t, p)
        union = np.logical_or(t, p)
        iou = (np.sum(intersection > 0) + 1e-10 )/ (np.sum(union > 0) + 1e-10)
        thresholds = np.arange(0.5, 1, 0.05)
        s = []
        for thresh in thresholds:
            s.append(iou > thresh)
        metric.append(np.mean(s))

    return np.mean(metric)

def my_iou_metric(label, pred):
    return tf.py_func(get_iou_vector, [label, pred>0.5], tf.float64)

def my_iou_metric_2(label, pred):
    return tf.py_func(get_iou_vector, [label, pred >0], tf.float64)


# code download from: https://github.com/bermanmaxim/LovaszSoftmax
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred = K.cast(y_pred, 'float32')
    y_pred_f = K.cast(K.greater(K.flatten(y_pred), 0.5), 'float32')
    intersection = y_true_f * y_pred_f
    score = 2. * K.sum(intersection) / (K.sum(y_true_f) + K.sum(y_pred_f))
    return score

def dice_loss(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = y_true_f * y_pred_f
    score = (2. * K.sum(intersection) + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return 1. - score

from keras import backend as K

def bce_dice_loss(y_true, y_pred):
    return binary_crossentropy(y_true, K.sigmoid(y_pred)) + dice_loss(y_true, K.sigmoid(y_pred))

def bce_logdice_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) - K.log(1. - dice_loss(y_true, y_pred))

def weighted_bce_loss(y_true, y_pred, weight):
    epsilon = 1e-7
    y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
    logit_y_pred = K.log(y_pred / (1. - y_pred))
    loss = weight * (logit_y_pred * (1. - y_true) + 
                     K.log(1. + K.exp(-K.abs(logit_y_pred))) + K.maximum(-logit_y_pred, 0.))
    return K.sum(loss) / K.sum(weight)

def weighted_dice_loss(y_true, y_pred, weight):
    smooth = 1.
    w, m1, m2 = weight, y_true, y_pred
    intersection = (m1 * m2)
    score = (2. * K.sum(w * intersection) + smooth) / (K.sum(w * m1) + K.sum(w * m2) + smooth)
    loss = 1. - K.sum(score)
    return loss

def weighted_bce_dice_loss(y_true, y_pred):
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')
    # if we want to get same size of output, kernel size must be odd
    averaged_mask = K.pool2d(
            y_true, pool_size=(50, 50), strides=(1, 1), padding='same', pool_mode='avg')
    weight = K.ones_like(averaged_mask)
    w0 = K.sum(weight)
    weight = 5. * K.exp(-5. * K.abs(averaged_mask - 0.5))
    w1 = K.sum(weight)
    weight *= (w0 / w1)
    loss = weighted_bce_loss(y_true, y_pred, weight) + dice_loss(y_true, y_pred)
    return loss


"""
Lovasz-Softmax and Jaccard hinge loss in Tensorflow
Maxim Berman 2018 ESAT-PSI KU Leuven (MIT License)
"""


import tensorflow as tf
import numpy as np

def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    gts = tf.reduce_sum(gt_sorted)
    intersection = gts - tf.cumsum(gt_sorted)
    union = gts + tf.cumsum(1. - gt_sorted)
    jaccard = 1. - intersection / union
    jaccard = tf.concat((jaccard[0:1], jaccard[1:] - jaccard[:-1]), 0)
    return jaccard


# --------------------------- BINARY LOSSES ---------------------------


def lovasz_hinge(logits, labels, per_image=True, ignore=None):
    """
    Binary Lovasz hinge loss
      logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
      labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
      per_image: compute the loss per image instead of per batch
      ignore: void class id
    """
    if per_image:
        def treat_image(log_lab):
            log, lab = log_lab
            log, lab = tf.expand_dims(log, 0), tf.expand_dims(lab, 0)
            log, lab = flatten_binary_scores(log, lab, ignore)
            return lovasz_hinge_flat(log, lab)
        losses = tf.map_fn(treat_image, (logits, labels), dtype=tf.float32)
        loss = tf.reduce_mean(losses)
    else:
        loss = lovasz_hinge_flat(*flatten_binary_scores(logits, labels, ignore))
    return loss


def lovasz_hinge_flat(logits, labels):
    """
    Binary Lovasz hinge loss
      logits: [P] Variable, logits at each prediction (between -\infty and +\infty)
      labels: [P] Tensor, binary ground truth labels (0 or 1)
      ignore: label to ignore
    """

    def compute_loss():
        labelsf = tf.cast(labels, logits.dtype)
        signs = 2. * labelsf - 1.
        errors = 1. - logits * tf.stop_gradient(signs)
        errors_sorted, perm = tf.nn.top_k(errors, k=tf.shape(errors)[0], name="descending_sort")
        gt_sorted = tf.gather(labelsf, perm)
        grad = lovasz_grad(gt_sorted)
        loss = tf.tensordot(tf.nn.relu(errors_sorted), tf.stop_gradient(grad), 1, name="loss_non_void")
        return loss

    # deal with the void prediction case (only void pixels)
    loss = tf.cond(tf.equal(tf.shape(logits)[0], 0),
                   lambda: tf.reduce_sum(logits) * 0.,
                   compute_loss,
                   strict=True,
                   name="loss"
                   )
    return loss


def flatten_binary_scores(scores, labels, ignore=None):
    """
    Flattens predictions in the batch (binary case)
    Remove labels equal to 'ignore'
    """
    scores = tf.reshape(scores, (-1,))
    labels = tf.reshape(labels, (-1,))
    if ignore is None:
        return scores, labels
    valid = tf.not_equal(labels, ignore)
    vscores = tf.boolean_mask(scores, valid, name='valid_scores')
    vlabels = tf.boolean_mask(labels, valid, name='valid_labels')
    return vscores, vlabels


# --------------------------- MULTICLASS LOSSES ---------------------------


def lovasz_softmax(probas, labels, classes='all', per_image=False, ignore=None, order='BHWC'):
    """
    Multi-class Lovasz-Softmax loss
      probas: [B, H, W, C] or [B, C, H, W] Variable, class probabilities at each prediction (between 0 and 1)
      labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
      per_image: compute the loss per image instead of per batch
      ignore: void class labels
      order: use BHWC or BCHW
    """
    if per_image:
        def treat_image(prob_lab):
            prob, lab = prob_lab
            prob, lab = tf.expand_dims(prob, 0), tf.expand_dims(lab, 0)
            prob, lab = flatten_probas(prob, lab, ignore, order)
            return lovasz_softmax_flat(prob, lab, classes=classes)
        losses = tf.map_fn(treat_image, (probas, labels), dtype=tf.float32)
        loss = tf.reduce_mean(losses)
    else:
        loss = lovasz_softmax_flat(*flatten_probas(probas, labels, ignore, order), classes=classes)
    return loss


def lovasz_softmax_flat(probas, labels, classes='all'):
    """
    Multi-class Lovasz-Softmax loss
      probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
      labels: [P] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
    """
    C = 1
    losses = []
    present = []
    class_to_sum = list(range(C)) if classes in ['all', 'present'] else classes
    for c in class_to_sum:
        fg = tf.cast(tf.equal(labels, c), probas.dtype)  # foreground for class c
        if classes == 'present':
            present.append(tf.reduce_sum(fg) > 0)
        errors = tf.abs(fg - probas[:, c])
        errors_sorted, perm = tf.nn.top_k(errors, k=tf.shape(errors)[0], name="descending_sort_{}".format(c))
        fg_sorted = tf.gather(fg, perm)
        grad = lovasz_grad(fg_sorted)
        losses.append(
            tf.tensordot(errors_sorted, tf.stop_gradient(grad), 1, name="loss_class_{}".format(c))
                      )
    if len(class_to_sum) == 1:  # short-circuit mean when only one class
        return losses[0]
    losses_tensor = tf.stack(losses)
    if classes == 'present':
        present = tf.stack(present)
        losses_tensor = tf.boolean_mask(losses_tensor, present)
    loss = tf.reduce_mean(losses_tensor)
    return loss


def flatten_probas(probas, labels, ignore=None, order='BHWC'):
    """
    Flattens predictions in the batch
    """
    if order == 'BCHW':
        probas = tf.transpose(probas, (0, 2, 3, 1), name="BCHW_to_BHWC")
        order = 'BHWC'
    if order != 'BHWC':
        raise NotImplementedError('Order {} unknown'.format(order))
    C = 1
    probas = tf.reshape(probas, (-1, C))
    labels = tf.reshape(labels, (-1,))
    if ignore is None:
        return probas, labels
    valid = tf.not_equal(labels, ignore)
    vprobas = tf.boolean_mask(probas, valid, name='valid_probas')
    vlabels = tf.boolean_mask(labels, valid, name='valid_labels')
    return vprobas, vlabels

def keras_lovasz_softmax(labels,probas):
    #return lovasz_softmax(probas, labels)+binary_crossentropy(labels, probas)
    return lovasz_softmax(probas, labels)

def keras_lovasz_hinge(labels,logits):
    return lovasz_hinge(logits, labels, per_image=True, ignore=None)

def get_data(word, df):
    a = np.zeros((len(df),net_size,net_size,3))
    for i in range(len(df)):
        tmp =  cv2.imread('train/{}/{}'.format(word,df[0][i]))
        a[i] = upsample(tmp)
    return data

def lovasz_loss(y_true, y_pred):
    y_true, y_pred = K.cast(K.squeeze(y_true, -1), 'int32'), K.cast(K.squeeze(y_pred, -1), 'float32')
    #logits = K.log(y_pred / (1. - y_pred))
    logits = y_pred #Jiaxin
    loss = lovasz_hinge(logits, y_true, per_image = True, ignore = None)
    return loss



X_train = np.array([upsample(cv2.imread('train/images/{}'.format(name))) for name in train_df[0]],np.float32) / 255
Y_train = np.array([upsample(cv2.imread('train/masks/{}'.format(name))) for name in train_df[0]],np.float32) / 255
X_val   = np.array([upsample(cv2.imread('train/images/{}'.format(name))) for name in val_df[0]],np.float32) / 255
Y_val   = np.array([upsample(cv2.imread('train/masks/{}'.format(name))) for name in val_df[0]],np.float32) / 255

X_train = np.append(X_train, [np.fliplr(x) for x in X_train], axis=0)
Y_train = np.append(Y_train, [np.fliplr(x) for x in Y_train], axis=0)

X_test =  np.array([upsample(cv2.imread('test/images/{}'.format(name))) for name in test_df[0]],np.float32) / 255
X_TTA = np.array([np.fliplr(x) for x in X_test])



    
MODEL_NAME = 'FOLDS_resnet34'
epochs = 130
batch_size = 64

X_train_splitted = np.array(np.array_split(X_train, 5))
y_train_splitted = np.array(np.array_split(Y_train, 5))

def get_model():
    input_layer = Input((net_size, net_size, 1))
    output_layer = build_model(input_layer, 32,0.5)
    return Model(input_layer,output_layer)



epochs = 130
batch_size = 60




X_train_splitted = np.array(np.array_split(X_train, 5))
y_train_splitted = np.array(np.array_split(Y_train, 5))

def get_model():
    input_layer = Input((net_size, net_size, 1))
    output_layer = build_model(input_layer, 32,0.5)
    return Model(input_layer,output_layer)
    
models = [get_model() for i in range(5)]
model1 = models[0]
MODEL_NAME = 'FOLDS_resnet34'
#c = optimizers.adam(lr = 0.001)
for model in models:
    model.compile(loss="binary_crossentropy", optimizer='adam', metrics=[my_iou_metric])
    
MODEL_NAME = 'FOLDS_resnet34'
epochs = 35
batch_size = 64

X_train_splitted = np.array(np.array_split(X_train, 5))
y_train_splitted = np.array(np.array_split(Y_train, 5))

for i in range(5):
    
    train_x         =  X_train_splitted[(np.arange(5)!=i)].reshape((4*1440,128,128,3))
    train_y         =  y_train_splitted[(np.arange(5)!=i)].reshape((4*1440,128,128,3))
    val_x,val_y     = X_train_splitted[i],y_train_splitted[i]
    
    
    callbacks = [
             ModelCheckpoint(monitor=' val_my_iou_metric',
                             filepath='weights/' + str(i) + '_' + MODEL_NAME,
                             save_best_only=True,
                             save_weights_only=True),
            LRTensorBoard(log_dir='./logs/{}'.format(str(i) + '_' +MODEL_NAME)),
            SGDRScheduler(min_lr=1e-6,
                                     max_lr=1e-3,
                                     steps_per_epoch=np.ceil(float(4*1440) / float(batch_size)),
                                     lr_decay=0.9,
                                     cycle_length=10,
                                     mult_factor=1.5)
            ]
    
    model = models[i]

    
    model.fit(train_x[:,:,:,:1],train_y[:,:,:,:1],validation_data=[val_x[:,:,:,:1], val_y[:,:,:,:1]], epochs=epochs, batch_size=batch_size, callbacks=callbacks)
    model.save('folds_s/{}.h5'.format(str(i) + '_' +MODEL_NAME))
    

MODEL_NAME = 'FOLDS_resnet34'
from keras import optimizers
for i in range(5):
    models[i] = Model(models[i].layers[0].input,[models[i].layers[-1].input,models[i].layers[-1].output])
    models[i].compile(loss=[lovasz_loss,weighted_bce_dice_loss], optimizer='sgd', loss_weights=[0.9,0.1],metrics=[my_iou_metric_2])

for i in range(5):
    
    train_x         =  X_train_splitted[(np.arange(5)!=i)].reshape((4*1440,128,128,3))
    train_y         =  y_train_splitted[(np.arange(5)!=i)].reshape((4*1440,128,128,3))
    val_x,val_y     =  X_train_splitted[i],y_train_splitted[i]
    
    
    callbacks = [
             ModelCheckpoint(monitor='val_my_iou_metric_2',
                             filepath='weights/lavaz_' + str(i) + '_' + MODEL_NAME,
                             save_best_only=True,
                             save_weights_only=True),
            LRTensorBoard(log_dir='./logs/lavaz_{}'.format(str(i) + '_' +MODEL_NAME)),
            SGDRScheduler(min_lr=1e-6,
                                     max_lr=1e-2,
                                     steps_per_epoch=np.ceil(float(4*1440) / float(batch_size)),
                                     lr_decay=0.9,
                                     cycle_length=10,
                                     mult_factor=1.5)
            ]
    
    model = models[i]
    
    model.fit(train_x[:,:,:,:1],[train_y[:,:,:,:1],train_y[:,:,:,:1]],validation_data=[val_x[:,:,:,:1], val_y[:,:,:,:1], val_y[:,:,:,:1]], epochs=130, batch_size=batch_size, callbacks=callbacks)
    model.save('folds_s/2_bce_lavaz_{}.h5'.format(str(i) + '_' +MODEL_NAME))


MODEL_NAME = 'FOLDS_resnet34'

#models = []
for i in range(5):
#    models.append(load_model('folds_s/bce_dice_{}.h5'.format(str(i) + '_' +MODEL_NAME), custom_objects={'my_iou_metric': my_iou_metric,'weighted_bce_dice_loss':weighted_bce_dice_loss}))
#    models.append()
    for j in range(len(models[i].layers)):
        models[i].layers[j].name='{}_{}'.format(str(i),str(j))

conc = concatenate([model.output[1] for model in models])
out = Conv2D(1,8,padding='same', activation='sigmoid')(conc)
model = Model([model.input for model in models],out)
model.compile(loss=weighted_bce_dice_loss, optimizer='sgd', metrics=[my_iou_metric])
for layer in model.layers:
    layer.trainable = False
model.layers[-1].trainable = True
model.layers[-2].trainable = True

epochs = 70
batch_size = 15

callbacks = [
             ModelCheckpoint(monitor=' val_my_iou_metric_2',
                             filepath='weights/to_pro_lavaz__' + '_' + MODEL_NAME,
                             save_best_only=True,
                             save_weights_only=True),
            LRTensorBoard(log_dir='./logs/to_pro_bcedice_{}'.format('_' +MODEL_NAME)),
            SGDRScheduler(min_lr=1e-6,
                                     max_lr=1e-3,
                                     steps_per_epoch=np.ceil(float(5*1440) / float(batch_size)),
                                     lr_decay=0.9,
                                     cycle_length=10,
                                     mult_factor=1.5)
            ]
# X = np.array([X_train[:,:,:,:1] for i in range(5)],np.float32)
# X_val = np.array([X_val[:,:,:,:1] for i in range(5)],np.float32)
X = []
X_v = []
for i in range(5):
    
    X.append(X_train[:,:,:,:1])
    X_v.append(X_val[:,:,:,:1])
model.fit(X,Y_train[:,:,:,:1],validation_data=[X_v, Y_val[:,:,:,:1]], epochs=epochs, batch_size=batch_size, callbacks=callbacks)
model.save('folds_s/to_pro_lavaz__{}.h5'.format(str(i) + '_' +MODEL_NAME))


def RLenc(img, order='F', format=True):
    """
    img is binary mask image, shape (r,c)
    order is down-then-right, i.e. Fortran
    format determines if the order needs to be preformatted (according to submission rules) or not

    returns run length as an array or string (if format is True)
    """
    bytes = img.reshape(img.shape[0] * img.shape[1], order=order)
    runs = []  ## list of run lengths
    r = 0  ## the current run length
    pos = 1  ## count starts from 1 per WK
    for c in bytes:
        if (c == 0):
            if r != 0:
                runs.append((pos, r))
                pos += r
                r = 0
            pos += 1
        else:
            r += 1

    # if last run is unsaved (i.e. data ends with 1)
    if r != 0:
        runs.append((pos, r))
        pos += r
        r = 0

    if format:
        z = ''

        for rr in runs:
            z += '{} {} '.format(rr[0], rr[1])
        return z[:-1]
    else:
        return runs

path_test = './test/'

depths = pd.read_csv('depths.csv', index_col='id')
test_ids = next(os.walk(path_test+"images"))[2]

X_test = np.zeros((len(test_ids), 128, 128, 3), dtype=np.uint8)
X_test_tta = np.zeros((len(test_ids), 128, 128, 3), dtype=np.uint8)
print('Getting and resizing test images ... ')
for n, id_ in (enumerate(test_ids)):
    path = path_test
    img = cv2.imread(path + 'images/' + id_)
    #x = img_to_array(img)[:,:,1]
    x = upsample(img)
    X_test[n] = x
    X_test_tta[n] = np.fliplr(x)
    
X_test = np.array(X_test,np.float32) / 255
X_test_tta = np.array(X_test_tta, np.float32) / 255

preds_test = model.predict([X_test[:,:,:,:1],X_test[:,:,:,:1],X_test[:,:,:,:1],X_test[:,:,:,:1]], verbose=1)
preds_test_tta = model.predict([X_test_tta[:,:,:,:1],X_test_tta[:,:,:,:1],X_test_tta[:,:,:,:1],X_test_tta[:,:,:,:1]], verbose=1)
preds_tmp = np.array([np.fliplr(y) for y in preds_test_tta])
preds_test = (preds_test + preds_tmp) /2
preds_test_t = (preds_test > 0.5).astype(np.uint8)
preds_test_upsampled = []
for i in range(len(preds_test)):
    preds_test_upsampled.append(downsample(preds_test_t[i]))
preds_test_upsampled = np.array(preds_test_upsampled)
pred_dict = {fn[:-4]:RLenc(np.round(preds_test_upsampled[i])) for i,fn in (enumerate(test_ids))}
sub = pd.DataFrame.from_dict(pred_dict,orient='index')
sub.index.names = ['id']
sub.columns = ['rle_mask']
sub.to_csv('submission.csv')
