import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers

import pickle
import numpy as np

from keras_preprocessing.sequence import pad_sequences
from util import preprocess

best_acc_list = [[4, 4, 4, 6, 5, 8, 6, 7, 3, 4],
[4, 4, 3, 3, 5, 8, 2, 4, 4, 4],
[7, 6, 7, 6, 5, 4, 7, 4, 6, 3],
[7, 5, 3, 6, 3, 4, 4, 3, 8, 3],
[10, 4, 4, 5, 5, 6, 7, 6, 7, 4],
[4, 3, 4, 8, 6, 5, 4, 5, 3, 4],
[6, 3, 5, 8, 5, 3, 5, 7, 8, 4],
[6, 5, 6, 5, 4, 3, 4, 3, 7, 4],
[7, 5, 4, 4, 2, 7, 6, 5, 6, 5],
[7, 5, 6, 4, 3, 7, 7, 3, 5, 4]
]
min_loss_list = [[4, 6, 8, 6, 7, 8, 6, 7, 5, 7],
[7, 5, 7, 6, 5, 8, 5, 5, 5, 4],    
[7, 6, 7, 6, 8, 7, 7, 5, 5, 6],
[7, 5, 8, 6, 5, 6, 4, 7, 8, 5],
[6, 5, 7, 5, 5, 6, 5, 6, 7, 5],
[5, 7, 5, 8, 6, 8, 5, 5, 4, 5],
[6, 6, 4, 8, 7, 8, 5, 6, 8, 7],
[7, 5, 6, 5, 8, 5, 4, 3, 5, 4],
[6, 3, 5, 4, 5, 5, 6, 5, 6, 5],
[7, 5, 4, 6, 5, 7, 5, 3, 5, 5]
]
# for i in range(0,10):
#     for j in range(0,10):
#         loaded_model = keras.models.load_model('models/10fold%d/bagging0/rnn_with_feature%d/model_%d'%(i,j,min_loss_list[i][j]))
#         loaded_model.save_weights('models/10fold%d/bagging0/rnn_with_feature%d/rnn_weights_min_loss%d.h5'%(i,j,j))


loaded_model = keras.models.load_model('models/10fold0/bagging1/rnn_with_feature5/model_2')
loaded_model.save_weights('models/10fold0/bagging1/rnn_with_feature5/rnn_weights_min_loss5.h5')

def multi_input_model(sequence_shape, image_shape, feature_shape, num_classes, cnn_lr_lamd=5):

    sequence_inputs = keras.Input(shape=sequence_shape, name='sequence')
    masking_layer = layers.Masking(mask_value=0, name='mask')
    x1 = masking_layer(sequence_inputs)
    x1 = layers.Bidirectional(layers.GRU(128, return_sequences=True, name='GRU1', trainable=False), name='Bidirectional_1',trainable=False)(x1)
    x1 = layers.Bidirectional(layers.GRU(64, name='GRU2', trainable=False), name='Bidirectional_2', trainable=False)(x1)
    x1 = layers.Dense(128, activation="relu", name='rnn_dense_1', trainable=False)(x1)
    x1 = layers.Dense(32, activation="relu", name='rnn_dense_2', trainable=False)(x1)

    image_inputs = keras.Input(shape=image_shape, name='image')
    x2 = layers.Rescaling(scale=1.0 / 255)(image_inputs)
    x2 = layers.Conv2D(filters=4, kernel_size=(7,7), strides=1, activation="relu", padding='same')(x2)
    x2 = layers.Conv2D(filters=4, kernel_size=(5,5), strides=1, activation="relu", padding='same')(x2)
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.Dropout(0.2)(x2)
    x2 = layers.MaxPooling2D(pool_size=(3, 3) ,strides=2)(x2)
    x2 = layers.Conv2D(filters=8, kernel_size=(3,3), padding='same', activation="relu")(x2)
    x2 = layers.Conv2D(filters=8, kernel_size=(3,3), padding='same', activation="relu")(x2)
    x2 = layers.Conv2D(filters=8, kernel_size=(3,3), padding='same', activation="relu")(x2)
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.Dropout(0.2)(x2)
    x2 = layers.MaxPooling2D(pool_size=(3, 3), strides=2)(x2)
    x2 = layers.Conv2D(filters=16, kernel_size=(3,3), padding='same', activation="relu")(x2)
    x2 = layers.Conv2D(filters=16, kernel_size=(3,3), padding='same', activation="relu")(x2)
    x2 = layers.Conv2D(filters=16, kernel_size=(3,3), padding='same', activation="relu")(x2)
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.Dropout(0.2)(x2)
    x2 = layers.MaxPooling2D(pool_size=(3, 3), strides=2)(x2)
    x2 = layers.Conv2D(filters=32, kernel_size=(3,3), padding='same', activation="relu")(x2)
    x2 = layers.Conv2D(filters=32, kernel_size=(3,3), padding='same', activation="relu")(x2)
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.Dropout(0.2)(x2)
    x2 = layers.MaxPooling2D(pool_size=(3, 3), strides=2)(x2)
    x2 = layers.GlobalAveragePooling2D()(x2)
    x2 = layers.Dropout(0.2)(x2)
    x2 = layers.Dense(256, activation="relu", kernel_regularizer='l1')(x2)
    x2 = layers.Dropout(0.2)(x2)
    x2 = layers.Dense(32, activation="relu", kernel_regularizer='l1')(x2)

    feature_inputs = keras.Input(shape=feature_shape, name='feature')
    x3 = feature_inputs
    x3 = layers.Dense(64, activation="relu", name='feature_dense_1', trainable=False)(x3)
    x3 = layers.Dense(32, activation="relu",  name='feature_dense_2', trainable=False)(x3)

    x = layers.concatenate([x1, x2, x3])
    outputs = layers.Dense(num_classes, activation="softmax", name='mix_final_dense')(x)
    model = keras.Model(inputs=[sequence_inputs, image_inputs, feature_inputs], outputs=outputs)
    return model

def build_channel(data):
    X = []
    for image in data:
        X.append(image.reshape(128, 128, 1))
    return X

def read_data(K=10, fold=0):

    f = open('data/split_%dfold_No%d'%(K,fold)+'_aug_to_2500_down_sample_True_instance0-9/val_data_image', 'rb')
    X_image_val = pickle.load(f)
    X_image_val = np.array(build_channel(X_image_val))
    f.close()

    f2 = open('data/split_%dfold_No%d'%(K,fold)+'_aug_to_2500_down_sample_True_instance0-9/val_data', 'rb')
    X_sequence_val, X_feature_val, Y_val = pickle.load(f2) 
    f2.close()

    total_X_sequence_train = []
    total_X_feature_train = []
    total_X_image_train = []
    total_Y_train = []

    for i in range(0,10):
        f = open('data/split_%dfold_No%d'%(K,fold)+'_aug_to_2500_down_sample_True_instance0-9/train_data%d'%i, 'rb')
        X_sequence_train, X_feature_train, Y_train = pickle.load(f)
        X_sequence_train_paded = pad_sequences(X_sequence_train, dtype='float', padding='post', value = np.nan)
        X_sequence_train_trimed = preprocess(X_sequence_train_paded)
        X_feature_train = np.array(X_feature_train)
        Y_train = np.array(Y_train)-1
        Y_train = keras.utils.to_categorical(Y_train, num_classes=11)
        total_X_sequence_train.append(X_sequence_train_trimed)
        total_X_feature_train.append(X_feature_train)
        total_Y_train.append(Y_train)
        f.close()

        f = open('data/split_%dfold_No%d'%(K,fold)+'_aug_to_2500_down_sample_True_instance0-9/train_data_image%d'%i, 'rb')
        X_image_train = pickle.load(f)
        X_image_train = np.array(build_channel(X_image_train))
        total_X_image_train.append(X_image_train)
        f.close()

    X_sequence_val_paded = pad_sequences(X_sequence_val, dtype='float', padding='post', value = np.nan)
    X_sequence_val_trimed = preprocess(X_sequence_val_paded)
    X_feature_val = np.array(X_feature_val)
    Y_val = np.array(Y_val)-1
    Y_val = keras.utils.to_categorical(Y_val, num_classes=11)

    return total_X_sequence_train, total_X_feature_train, total_X_image_train, total_Y_train, X_sequence_val_trimed, X_feature_val, X_image_val, Y_val

if __name__ == '__main__':
    print('strat')
    for fold in range(0,1):
        total_X_sequence_train, total_X_feature_train, total_X_image_train, total_Y_train , \
        X_sequence_val_trimed, X_feature_val, X_image_val, Y_val = read_data(fold=fold)
        X_val = {'sequence':X_sequence_val_trimed, 'feature':X_feature_val, 'image':X_image_val}
        for i in range(5,6):
            model = multi_input_model(sequence_shape=(None, 3), image_shape=(128, 128, 1), feature_shape=(2,), num_classes=11)
            model.compile(
            optimizer=keras.optimizers.Adam(1e-3),
            loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
            metrics=[keras.metrics.CategoricalAccuracy(name="acc")],
            run_eagerly=False,
            )
            callbacks = [
            keras.callbacks.ModelCheckpoint(
                filepath='models/10fold%d/bagging_image/bagging_after_image%d/model_{epoch}'%(fold,i),
                save_freq='epoch'),
            keras.callbacks.TensorBoard(log_dir='models/10fold%d/bagging_image/bagging_after_image%d/logs'%(fold,i))
            ]
            X_train = {'sequence':total_X_sequence_train[i], 'feature':total_X_feature_train[i], 'image':total_X_image_train[i]}
            model.load_weights('models/10fold%d/bagging1/rnn_with_feature%d/rnn_weights_min_loss%d.h5'%(fold,i,i), by_name=True)
            model.fit(X_train, total_Y_train[i], batch_size=32, epochs=100, 
                validation_data=(X_val, Y_val), callbacks=callbacks)