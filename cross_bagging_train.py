import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers
import pickle
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from util import preprocess

def read_data(K=10, fold=0):
    f2 = open('data/split_%dfold_No%d'%(K,fold)+'_aug_to_2500_down_sample_True_instance0-9/val_data', 'rb')
    X_sequence_val, X_feature_val, Y_val = pickle.load(f2) 
    f2.close()

    total_X_sequence_train = []
    total_X_feature_train = []
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

    X_sequence_val_paded = pad_sequences(X_sequence_val, dtype='float', padding='post', value = np.nan)
    X_sequence_val_trimed = preprocess(X_sequence_val_paded)
    X_feature_val = np.array(X_feature_val)
    Y_val = np.array(Y_val)-1
    Y_val = keras.utils.to_categorical(Y_val, num_classes=11)

    return total_X_sequence_train, total_X_feature_train, total_Y_train, X_sequence_val_trimed, X_feature_val, Y_val

def rnn_with_feature(sequence_shape, feature_shape, num_classes):
    sequence_inputs = keras.Input(shape=sequence_shape, name='sequence')
    masking_layer = layers.Masking(mask_value=0)
    x1 = masking_layer(sequence_inputs)
    x1 = layers.Bidirectional(layers.GRU(128, return_sequences=True))(x1)
    x1 = layers.Bidirectional(layers.GRU(64))(x1)
    x1 = layers.Dropout(0.4)(x1)
    x1 = layers.Dense(128, activation="relu")(x1)
    x1 = layers.Dropout(0.4)(x1)
    x1 = layers.Dense(32, activation="relu")(x1)

    feature_inputs = keras.Input(shape=feature_shape, name='feature')
    x3 = feature_inputs
    x3 = layers.Dense(64, activation="relu")(x3)
    x3 = layers.Dense(32, activation="relu")(x3)

    x = layers.concatenate([x1, x3])

    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inputs=[sequence_inputs, feature_inputs], outputs=outputs)
    return model

if __name__ == '__main__':
    print('strat')

    # lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    # initial_learning_rate=1e-3,
    # decay_steps=430,
    # decay_rate=0.9)

    # boundaries = [2000]
    # values = [1e-3, 1e-4]
    # lr_schedule= tf.keras.optimizers.schedules.PiecewiseConstantDecay(
    # boundaries, values, name=None
    # )

    for fold in range(1,10):
        total_X_sequence_train, total_X_feature_train, total_Y_train , \
        X_sequence_val_trimed, X_feature_val, Y_val = read_data(fold=fold)
        X_val = {'sequence':X_sequence_val_trimed, 'feature':X_feature_val}
        for i in range(0,10):
            model = rnn_with_feature(sequence_shape=(None, 3),feature_shape=(2,), num_classes=11)
            model.compile(
            optimizer=keras.optimizers.Adam(1e-3),
            loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
            metrics=[keras.metrics.CategoricalAccuracy(name="acc")],
            )
            callbacks = [
            keras.callbacks.ModelCheckpoint(
                filepath='models/10fold%d/bagging0/rnn_with_feature%d/model_{epoch}'%(fold,i),
                save_freq='epoch'),
            keras.callbacks.TensorBoard(log_dir='models/10fold%d/bagging0/rnn_with_feature%d/logs'%(fold,i))
            ]
            X_train = {'sequence':total_X_sequence_train[i], 'feature':total_X_feature_train[i]}
            model.fit(X_train, total_Y_train[i], batch_size=32, epochs=10, 
                validation_data=(X_val, Y_val), callbacks=callbacks)