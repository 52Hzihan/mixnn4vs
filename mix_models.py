import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers

def rnn_with_feature(sequence_shape, feature_shape, num_classes, dropout_rate=0.4):
    sequence_inputs = keras.Input(shape=sequence_shape, name='sequence')
    masking_layer = layers.Masking(mask_value=0)
    x1 = masking_layer(sequence_inputs)
    x1 = layers.Bidirectional(layers.GRU(128, return_sequences=True))(x1)
    x1 = layers.Bidirectional(layers.GRU(64))(x1)
    x1 = layers.Dropout(dropout_rate)(x1)
    x1 = layers.Dense(128, activation="relu")(x1)
    x1 = layers.Dropout(dropout_rate)(x1)
    x1 = layers.Dense(32, activation="relu")(x1)

    feature_inputs = keras.Input(shape=feature_shape, name='feature')
    x3 = feature_inputs
    x3 = layers.Dense(64, activation="relu")(x3)
    x3 = layers.Dense(32, activation="relu")(x3)

    x = layers.concatenate([x1, x3])

    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inputs=[sequence_inputs, feature_inputs], outputs=outputs)
    return model


def cnn_with_feature_1(image_shape, feature_shape, num_classes):

    image_inputs = keras.Input(shape=image_shape, name='image')

    x2 = layers.Rescaling(scale=1.0 / 255)(image_inputs)
    x2 = layers.Conv2D(filters=32, kernel_size=(16,16), strides=2, activation="relu", padding='same')(x2)
    x2 = layers.Dropout(0.2)(x2)
    x2 = layers.MaxPooling2D(pool_size=(3, 3) ,strides=2)(x2)
    x2 = layers.Conv2D(filters=64, kernel_size=(8,8), padding='same', activation="relu")(x2)
    x2 = layers.Dropout(0.2)(x2)
    x2 = layers.MaxPooling2D(pool_size=(3, 3), strides=2)(x2)
    x2 = layers.Conv2D(filters=64, kernel_size=(4,4), padding='same', activation="relu")(x2)
    x2 = layers.Dropout(0.2)(x2)
    x2 = layers.MaxPooling2D(pool_size=(3, 3), strides=2)(x2)
    x2 = layers.Conv2D(filters=128, kernel_size=(3,3), padding='same', activation="relu")(x2)
    x2 = layers.Dropout(0.2)(x2)
    x2 = layers.MaxPooling2D(pool_size=(3, 3), strides=2)(x2)
    x2 = layers.GlobalAveragePooling2D()(x2)
    x2 = layers.Dropout(0.2)(x2)
    x2 = layers.Dense(256, activation="relu", kernel_regularizer='l1')(x2)
    x2 = layers.Dropout(0.2)(x2)
    x2 = layers.Dense(32, activation="relu", kernel_regularizer='l1')(x2)

    feature_inputs = keras.Input(shape=feature_shape, name='feature')
    x3 = layers.Dense(64, activation="relu", kernel_regularizer='l1')(feature_inputs)
    x3 = layers.Dense(32, activation="relu", kernel_regularizer='l1')(x3)

    x = layers.concatenate([x2, x3])
    outputs = layers.Dense(num_classes, activation="softmax", kernel_regularizer='l1')(x)
    model = keras.Model(inputs=[image_inputs, feature_inputs], outputs=outputs)
    return model


def cnn_with_feature_2(image_shape, feature_shape, num_classes):
    
    image_inputs = keras.Input(shape=image_shape, name='image')

    x2 = layers.Rescaling(scale=1.0 / 255)(image_inputs)
    x2 = layers.Conv2D(filters=8, kernel_size=(3,3), strides=2, activation="relu", padding='same')(x2)
    x2 = layers.Conv2D(filters=8, kernel_size=(3,3), strides=2, activation="relu", padding='same')(x2)
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.Dropout(0.2)(x2)
    x2 = layers.MaxPooling2D(pool_size=(3, 3) ,strides=2)(x2)
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
    x3 = layers.Dense(64, activation="relu", kernel_regularizer='l1')(feature_inputs)
    x3 = layers.Dense(32, activation="relu", kernel_regularizer='l1')(x3)

    x = layers.concatenate([x2, x3])
    outputs = layers.Dense(num_classes, activation="softmax", kernel_regularizer='l1')(x)
    model = keras.Model(inputs=[image_inputs, feature_inputs], outputs=outputs)
    return model

def multi_input_model(sequence_shape, image_shape, feature_shape, num_classes):

    sequence_inputs = keras.Input(shape=sequence_shape, name='sequence')
    masking_layer = layers.Masking(mask_value=0)
    x1 = masking_layer(sequence_inputs)
    x1 = layers.Bidirectional(layers.GRU(128, return_sequences=True))(x1)
    x1 = layers.Bidirectional(layers.GRU(64))(x1)
    x1 = layers.Dropout(0.5)(x1)
    x1 = layers.Dense(128, activation="relu")(x1)
    x1 = layers.Dropout(0.5)(x1)
    x1 = layers.Dense(32, activation="relu")(x1)

    image_inputs = keras.Input(shape=image_shape, name='image')
    x2 = layers.Rescaling(scale=1.0 / 255)(image_inputs)
    x2 = layers.Conv2D(filters=32, kernel_size=(16,16), strides=2, activation="relu", padding='same')(x2)
    x2 = layers.Dropout(0.2)(x2)
    x2 = layers.MaxPooling2D(pool_size=(3, 3) ,strides=2)(x2)
    x2 = layers.Conv2D(filters=64, kernel_size=(8,8), padding='same', activation="relu")(x2)
    x2 = layers.Dropout(0.2)(x2)
    x2 = layers.MaxPooling2D(pool_size=(3, 3), strides=2)(x2)
    x2 = layers.Conv2D(filters=64, kernel_size=(4,4), padding='same', activation="relu")(x2)
    x2 = layers.Dropout(0.2)(x2)
    x2 = layers.MaxPooling2D(pool_size=(3, 3), strides=2)(x2)
    x2 = layers.Conv2D(filters=128, kernel_size=(3,3), padding='same', activation="relu")(x2)
    x2 = layers.Dropout(0.2)(x2)
    x2 = layers.MaxPooling2D(pool_size=(3, 3), strides=2)(x2)
    x2 = layers.GlobalAveragePooling2D()(x2)
    x2 = layers.Dropout(0.2)(x2)
    x2 = layers.Dense(256, activation="relu", kernel_regularizer='l1')(x2)
    x2 = layers.Dropout(0.2)(x2)
    x2 = layers.Dense(32, activation="relu", kernel_regularizer='l1')(x2)

    feature_inputs = keras.Input(shape=feature_shape, name='feature')
    x3 = layers.Dense(64, activation="relu")(feature_inputs)
    x3 = layers.Dense(32, activation="relu")(x3)

    x = layers.concatenate([x1, x2, x3])
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inputs=[sequence_inputs, image_inputs, feature_inputs], outputs=outputs)
    return model

def cnn(image_shape,  num_classes):
    image_inputs = keras.Input(shape=image_shape, name='image')
    x2 = layers.Rescaling(scale=1.0 / 255)(image_inputs)
    x2 = layers.Conv2D(filters=32, kernel_size=(16,16), strides=2, activation="relu", padding='same')(x2)
    x2 = layers.Dropout(0.2)(x2)
    x2 = layers.MaxPooling2D(pool_size=(3, 3) ,strides=2)(x2)
    x2 = layers.Conv2D(filters=64, kernel_size=(8,8), padding='same', activation="relu")(x2)
    x2 = layers.Dropout(0.2)(x2)
    x2 = layers.MaxPooling2D(pool_size=(3, 3), strides=2)(x2)
    x2 = layers.Conv2D(filters=64, kernel_size=(4,4), padding='same', activation="relu")(x2)
    x2 = layers.Dropout(0.2)(x2)
    x2 = layers.MaxPooling2D(pool_size=(3, 3), strides=2)(x2)
    x2 = layers.Conv2D(filters=128, kernel_size=(3,3), padding='same', activation="relu")(x2)
    x2 = layers.Dropout(0.2)(x2)
    x2 = layers.MaxPooling2D(pool_size=(3, 3), strides=2)(x2)
    x2 = layers.GlobalAveragePooling2D()(x2)
    x2 = layers.Dropout(0.2)(x2)
    x2 = layers.Dense(256, activation="relu", kernel_regularizer='l1')(x2)
    x2 = layers.Dropout(0.2)(x2)
    x2 = layers.Dense(128, activation="relu", kernel_regularizer='l1')(x2)
    x2 = layers.Dropout(0.2)(x2)
    outputs = layers.Dense(num_classes, activation="softmax", kernel_regularizer='l1')(x2)
    model = keras.Model(inputs=image_inputs, outputs=outputs)
    return model