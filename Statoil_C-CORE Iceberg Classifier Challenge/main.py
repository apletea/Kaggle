import shutil
import matplotlib.pyplot as plt
from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, Activation
from keras.models import Model, load_model
from keras.applications.vgg19 import preprocess_input,decode_predictions
from sklearn.model_selection import ParameterGrid, StratifiedKFold
import numpy as np
from keras_tqdm import TQDMNotebookCallback
from keras.callbacks import ModelCheckpoint, EarlyStopping
from tqdm import tqdm_notebook
from keras.preprocessing.image import ImageDataGenerator
def get_VGG19():

    model = VGG19(weights='imagenet',include_top=False,input_shape=(75, 75, 3))

    x = model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='selu')(x)
    x = Dropout(0.5)(x)
    x = Dense(64)(x)
    x = Activation('selu')(x)
    x = Dropout(0.5)(x)

    predictions = Dense(1, activation='sigmoid')(x)

    return Model(inputs=model.input, outputs=predictions)


def pop(self):
    '''Removes a layer instance on top of the layer stack.'''
    if not self.outputs:
        raise Exception('Sequential model cannot be popped: model is empty.')
    else:
        self.layers.pop()
        if not self.layers:
            self.outputs = []
            self.inbound_nodes = []
            self.outbound_nodes = []
        else:
            self.layers[-1].outbound_nodes = []
            self.outputs = [self.layers[-1].output]
        self.built = False


def pop_last_3_layers(model):
    pop(model)
    pop(model)
    pop(model)
    return model

def export_weight(model,filepath):
    model.save_weights(filepath)

def import_weight(model, filepath):
    model.weights(filepath, by_name=False)


def calc_mean_std(X):
    mean, std = np.zeros((3,)), np.zeros((3,))

    for x in X:
        x = x / 255.
        x = x.reshape(-1, 3)
        mean += x.mean(axis=0)
        std += x.std(axis=0)

    return mean / len(X), std / len(X)

def preprocess_poster(x, train_mean, train_std):
    x = x / 255.
    x -= train_mean
    x /= train_std
    return x

data = pd.read_json('fast.json')
data.inc_angle = data.inc_angle.apply(lambda x: np.nan if x == 'na' else x)
X_band_1=np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in data["band_1"]])
X_band_2=np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in data["band_2"]])
X_band_3=(X_band_1+X_band_2)/2
imgs =  np.concatenate([X_band_1[:, :, :, np.newaxis]
                          , X_band_2[:, :, :, np.newaxis]
                         , X_band_3[:, :, :, np.newaxis]], axis=-1)
X,y =data.drop(['is_iceberg','band_2','band_1','id'], axis=1).as_matrix(), data['is_iceberg'].values
print X
n_splits = (StratifiedKFold(n_splits=5))
n_splits.get_n_splits(X,y)
train_stat = []

batch_size = 32
checkpoints_prefix = './check/e/'

for fold_i, (train_idxs, test_idxs) in tqdm_notebook(list(enumerate(n_splits))):
    fold_stat = {}
    y_train, y_test = y[train_idxs], y[test_idxs]
    train_posters, test_posters = imgs[train_idxs], imgs[test_idxs]
    train_mean, train_std = calc_mean_std(train_posters)
    fold_stat['posters_train_mean'] = train_mean
    fold_stat['posters_train_std'] = train_std

    X_train_posters = np.array([preprocess_poster(poster_i, train_mean, train_std)
                               for poster_i in train_posters])

    train_datagen = ImageDataGenerator(
        shear_range=0.4,
        horizontal_flip=True,
        rotation_range=20.,
        width_shift_range=0.4,
        height_shift_range=0.4,
        zoom_range=0.4,
        vertical_flip=True,
    )

    train_datagen.fit(X_train_posters, augment=False)
    train_flow = train_datagen.flow(X_train_posters, y_train, batch_size=batch_size)

    X_test_posters = np.array([preprocess_poster(poster_i, train_mean, train_std)
                               for poster_i in test_posters])

    steps_per_epoch = len(X_train_posters) // batch_size
    model = get_VGG19()

    for layer in model.layers[:-5]:
        layer.trainable = False
    for layer in model.layers[-5:]:
        layer.trainable = True

    model.compile(optimizer='rmsprop', loss='binary_crossentropy',metrics=['accuracy'])
    base_checkpoints_path = os.path.join(checkpoints_prefix, 'base_checkpoints', str(fold_i))
    checkpoints_path = os.path.join(checkpoints_prefix, 'checkpoints', str(fold_i))


    shutil.rmtree(base_checkpoints_path, ignore_errors=True)
    shutil.rmtree(checkpoints_path, ignore_errors=True)

    os.makedirs(base_checkpoints_path)
    os.makedirs(checkpoints_path)

    base_checkpoints_path_template = os.path.join(base_checkpoints_path, 'base_checkpoint.hdf5')
    checkpoints_path_template = os.path.join(checkpoints_path, 'checkpoints.hdf5')

    base_checkpointer = ModelCheckpoint(base_checkpoints_path_template, save_best_only=True)
    checkpointer = ModelCheckpoint(checkpoints_path_template, save_best_only=True)

    base_history = model.fit_generator(
        train_flow,
        steps_per_epoch=steps_per_epoch,
        validation_data=(X_test_posters, y_test),
        epochs=10,
        callbacks=[TQDMNotebookCallback(leave_outer=False), base_checkpointer, EarlyStopping(patience=10)],
        verbose=0
    )

    fold_stat['base_history'] = base_history.history

    for layer in model.layers[:15]:
        layer.trainable = False
    for layer in model.layers[15:]:
        layer.trainable = True

