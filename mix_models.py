import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers

def rnn_with_feature(sequence_shape, feature_shape, num_classes, dropout_rate=0.4):
    sequence_inputs = keras.Input(shape=sequence_shape, name='sequence')
    masking_layer = layers.Masking(mask_value=0, name='mask')
    x1 = masking_layer(sequence_inputs)
    x1 = layers.Bidirectional(layers.GRU(128, return_sequences=True, name='GRU1'), name='Bidirectional_1')(x1)
    x1 = layers.Bidirectional(layers.GRU(64, name='GRU2'), name='Bidirectional_2')(x1)
    x1 = layers.Dropout(dropout_rate)(x1)
    x1 = layers.Dense(128, activation="relu", name='rnn_dense_1')(x1)
    x1 = layers.Dropout(dropout_rate)(x1)
    x1 = layers.Dense(32, activation="relu", name='rnn_dense_2')(x1)

    feature_inputs = keras.Input(shape=feature_shape, name='feature')
    x3 = feature_inputs
    x3 = layers.Dense(64, activation="relu", name='feature_dense_1')(x3)
    x3 = layers.Dense(32, activation="relu",  name='feature_dense_2')(x3)

    x = layers.concatenate([x1, x3])

    outputs = layers.Dense(num_classes, activation="softmax", name='rnn_final_dense')(x)
    model = keras.Model(inputs=[sequence_inputs, feature_inputs], outputs=outputs)
    return model

def rnn_with_feature_2(sequence_shape, feature_shape, num_classes, dropout_rate=0.4):
    sequence_inputs = keras.Input(shape=sequence_shape, name='sequence')
    masking_layer = layers.Masking(mask_value=0)
    x1 = masking_layer(sequence_inputs)
    x1 = layers.Bidirectional(layers.GRU(128, return_sequences=True))(x1)
    x1 = layers.Bidirectional(layers.GRU(64))(x1)
    x1 = layers.Dropout(dropout_rate)(x1)
    x1 = layers.Dense(128, activation="relu")(x1)
    x1 = layers.Dropout(dropout_rate)(x1)
    x1 = layers.Dense(32, activation="relu")(x1)
    x1 = layers.Dense(num_classes, activation="softmax")(x1)

    feature_inputs0 = keras.Input(shape=feature_shape, name='feature0')
    feature_inputs1 = keras.Input(shape=feature_shape, name='feature1')
    x3 = layers.Dense(num_classes, activation="softmax")(feature_inputs0)
    x4 = layers.Dense(num_classes, activation="softmax")(feature_inputs1)


    x = layers.concatenate([x1, x3, x4])
    x = layers.Dense(64, activation="relu")(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inputs=[sequence_inputs, feature_inputs0, feature_inputs1], outputs=outputs)
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
    x3 = layers.Dense(64, activation="relu", kernel_regularizer='l1')(feature_inputs)
    x3 = layers.Dense(32, activation="relu", kernel_regularizer='l1')(x3)

    x = layers.concatenate([x2, x3])
    outputs = layers.Dense(num_classes, activation="softmax", kernel_regularizer='l1')(x)
    model = keras.Model(inputs=[image_inputs, feature_inputs], outputs=outputs)
    return model

def cnn_with_feature_3(image_shape, feature_shape, num_classes):
    
    image_inputs = keras.Input(shape=image_shape, name='image')

    x2 = layers.Rescaling(scale=1.0 / 255)(image_inputs)
    x2 = layers.Conv2D(filters=8, kernel_size=(7,7), strides=1, activation="relu", padding='same')(x2)
    x2 = layers.Conv2D(filters=8, kernel_size=(5,5), strides=1, activation="relu", padding='same')(x2)
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.Dropout(0.2)(x2)
    x2 = layers.MaxPooling2D(pool_size=(3, 3) ,strides=2)(x2)
    x2 = layers.Conv2D(filters=16, kernel_size=(3,3), padding='same', activation="relu")(x2)
    x2 = layers.Conv2D(filters=16, kernel_size=(3,3), padding='same', activation="relu")(x2)
    x2 = layers.Conv2D(filters=16, kernel_size=(3,3), padding='same', activation="relu")(x2)
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
    x3 = layers.Dense(64, activation="relu", kernel_regularizer='l1')(feature_inputs)
    x3 = layers.Dense(32, activation="relu", kernel_regularizer='l1')(x3)

    x = layers.concatenate([x2, x3])
    outputs = layers.Dense(num_classes, activation="softmax", kernel_regularizer='l1')(x)
    model = keras.Model(inputs=[image_inputs, feature_inputs], outputs=outputs)
    return model

import keras.backend as K

class SetLearningRate:
    """层的一个包装，用来设置当前层的学习率
    """

    def __init__(self, layer, lamb, is_ada=False):
        self.layer = layer
        self.lamb = lamb # 学习率比例
        self.is_ada = is_ada # 是否自适应学习率优化器

    def __call__(self, inputs):
        with K.name_scope(self.layer.name):
            if not self.layer.built:
                input_shape = K.int_shape(inputs)
                self.layer.build(input_shape)
                self.layer.built = True
                if self.layer._initial_weights is not None:
                    self.layer.set_weights(self.layer._initial_weights)
        for key in ['kernel', 'bias', 'embeddings', 'depthwise_kernel', 'pointwise_kernel', 'recurrent_kernel', 'gamma', 'beta']:
            if hasattr(self.layer, key):
                weight = getattr(self.layer, key)
                if self.is_ada:
                    lamb = self.lamb # 自适应学习率优化器直接保持lamb比例
                else:
                    lamb = self.lamb**0.5 # SGD（包括动量加速），lamb要开平方
                K.set_value(weight, K.eval(weight) / lamb) # 更改初始化
                setattr(self.layer, key, weight * lamb) # 按比例替换
        return self.layer(inputs)

def multi_input_model(sequence_shape, image_shape, feature_shape, num_classes, cnn_lr_lamd=5):

    sequence_inputs = keras.Input(shape=sequence_shape, name='sequence')
    masking_layer = layers.Masking(mask_value=0, name='mask')
    x1 = masking_layer(sequence_inputs)
    x1 = layers.Bidirectional(layers.GRU(128, return_sequences=True, name='GRU1', trainable=False), name='Bidirectional_1',trainable=False)(x1)
    x1 = layers.Bidirectional(layers.GRU(64, name='GRU2', trainable=False), name='Bidirectional_2', trainable=False)(x1)
    x1 = layers.Dropout(0.4)(x1)
    x1 = layers.Dense(128, activation="relu", name='rnn_dense_1', trainable=False)(x1)
    x1 = layers.Dropout(0.4)(x1)
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
    x3 = layers.Dense(64, activation="relu", name='feature_dense_1', trainable=True)(x3)
    x3 = layers.Dense(32, activation="relu",  name='feature_dense_2', trainable=True)(x3)

    x = layers.concatenate([x1, x2, x3])
    outputs = layers.Dense(num_classes, activation="softmax", name='mix_final_dense')(x)
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