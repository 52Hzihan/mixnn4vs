import os
os.environ['CUDA_VISIBLE_DEVICES']='1'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers
import pickle
import numpy as np

from losses import categorical_focal_loss
from mix_models import rnn_with_feature


# for i in range(0, 10):
for i in (2,3,6,7,8):
    # f1 = open('data/split_10fold_No%d_aug_to_15000_down_sample_False_instance0-9/train_data1'%i, 'rb')
    f1 = open('data/split_10fold_No%d_aug_to_2500_down_sample_False_instance0-9/train_data1'%i, 'rb')
    X_sequence_train, X_feature_train, Y_train = pickle.load(f1)
    f1.close()
    # f2 = open('data/split_10fold_No%d_aug_to_15000_down_sample_False_instance0-9/val_data'%i, 'rb')
    f2 = open('data/split_10fold_No%d_aug_to_2500_down_sample_False_instance0-9/val_data'%i, 'rb') 
    X_sequence_val, X_feature_val,  Y_val = pickle.load(f2)
    f2.close()

    from keras.preprocessing.sequence import pad_sequences
    X_sequence_train_paded = pad_sequences(X_sequence_train, dtype='float', padding='post', value = np.nan)
    X_sequence_val_paded = pad_sequences(X_sequence_val, dtype='float', padding='post', value = np.nan)
    # 先pad再预处理，因为pad可以把array的list转为ndarray

    from util import preprocess
    X_sequence_train_trimed = preprocess(X_sequence_train_paded)
    X_sequence_val_trimed = preprocess(X_sequence_val_paded)


    X_feature_train = np.array(X_feature_train)
    X_feature_val = np.array(X_feature_val)
    Y_train = np.array(Y_train)-1 # 文件夹名从1开始但是标签从0开始
    Y_val = np.array(Y_val)-1

    Y_train_one_hot = keras.utils.to_categorical(Y_train, num_classes=11)
    Y_val_one_hot = keras.utils.to_categorical(Y_val, num_classes=11)

    f = open('data/split_10fold_No%d_aug_to_2500_down_sample_False_instance0-9/class_weights1'%i, 'rb')
    train_weight, test_weight = pickle.load(f)
    f.close()

    model = rnn_with_feature(sequence_shape=(None, 3),feature_shape=(2,), num_classes=11)
    # model.summary()
    # lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    #     initial_learning_rate=1e-3,
    #     decay_steps=860,
    #     decay_rate=0.85)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        # loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        loss=categorical_focal_loss(alpha=np.ones(11)),
        metrics=[keras.metrics.CategoricalAccuracy(name="acc")],
        run_eagerly=False,
    )
    callbacks = [
    keras.callbacks.ModelCheckpoint(
        # filepath='models/weighted_focal_loss_fold%d/model_{epoch}'%i,
        filepath='models/weighted_focal_loss_fold%d/model_{epoch}'%i,
        save_freq='epoch'),
    # keras.callbacks.TensorBoard(log_dir='models/weighted_focal_loss_fold%d/logs'%i)
    keras.callbacks.TensorBoard(log_dir='models/weighted_focal_loss_fold%d/logs'%i)
    ]

    X_train = {'sequence':X_sequence_train_trimed, 'feature':X_feature_train}
    X_val = {'sequence':X_sequence_val_trimed, 'feature':X_feature_val}
    model.fit(X_train, Y_train_one_hot, batch_size=32, epochs=20, 
            validation_data=(X_val, Y_val_one_hot), 
            class_weight=dict(enumerate(train_weight)),
            callbacks=callbacks )
