from keras.layers import Dense, MaxPooling2D, Flatten, Input, Convolution2D, BatchNormalization, Activation, \
    GaussianNoise, Dropout
from keras.models import Model
from keras.regularizers import l2

from KerasLayers.Custom_layers import LRN2D
from constants10 import *


def deep_model():
    inputs = Input(shape=(img_width, img_height, 3), name='input')
    if stddev > 0:
        inputs = GaussianNoise(stddev=stddev)(inputs)

    x = Convolution2D(32, (3, 3), border_mode='same', W_regularizer=l2(regularization), name='conv_1')(inputs)
    if batch_normalization:
        x = BatchNormalization(name='batch_norm_1')(x)
    x = Activation('relu')(x)

    x = MaxPooling2D(pool_size=(2, 2), name='max_pooling_1')(x)

    x = Convolution2D(64, (3, 3), border_mode='same', W_regularizer=l2(regularization), name='conv_2')(x)

    if batch_normalization:
        x = BatchNormalization(name='batch_norm_2')(x)
    x = Activation('relu')(x)

    x = MaxPooling2D((2, 2), name='max_pooling_2')(x)

    x = Convolution2D(64, (3, 3), border_mode='same', W_regularizer=l2(regularization), name='conv_3')(x)
    if batch_normalization:
        x = BatchNormalization(name='batch_norm_3')(x)
    x = Activation('relu')(x)

    x = MaxPooling2D((2, 2), name='max_pooling_3')(x)

    x = Flatten()(x)

    x = Dense(2048, activation='relu', W_regularizer=l2(regularization), name='fc1')(x)
    if dropout > 0:
        x = Dropout(dropout)(x)

    x = Dense(2048, activation='relu', W_regularizer=l2(regularization), name='fc2')(x)
    if dropout > 0:
        x = Dropout(dropout)(x)

    x = Dense(8, activation='softmax', name='predictions')(x)

    model = Model(inputs, x, name='deep_model')

    return model


def CNNS_model():
    if regularization > 0:
        W_regularizer = l2(regularization)
        b_regularizer = l2(regularization)
    else:
        W_regularizer = None
        b_regularizer = None

    # Input Block
    inputs = Input(shape=(img_width, img_height, 3), name='input')
    if stddev > 0:
        inputs = GaussianNoise(stddev=stddev)(inputs)

    x = Convolution2D(96, (7, 7), border_mode='same', subsample=2, W_regularizer=l2(regularization),
                      b_regularizer=l2(regularization), name='conv_1')(inputs)
    if LRN2D_norm:
        x = LRN2D(alpha=alpha, beta=beta)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(3, 3), name='max_pooling_1')(x)

    # Conv1 Block
    x = Convolution2D(256, (5, 5), border_mode='same', W_regularizer=W_regularizer,
                      b_regularizer=b_regularizer, name='conv_2')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), name='max_pooling_2')(x)

    # Conv2 Block
    x = Convolution2D(512, (3, 3), border_mode='same', W_regularizer=W_regularizer,
                      b_regularizer=b_regularizer, name='conv_3')(x)
    x = Activation('relu')(x)
    x = Convolution2D(512, (3, 3), border_mode='same', W_regularizer=W_regularizer,
                      b_regularizer=b_regularizer, name='conv_4')(x)
    x = Activation('relu')(x)
    x = Convolution2D(512, (3, 3), border_mode='same', W_regularizer=W_regularizer,
                      b_regularizer=b_regularizer, name='conv_5')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), name='max_pooling_3')(x)

    x = Flatten()(x)

    # Dense Block
    x = Dense(4096, activation='relu', name='full6')(x)
    if dropout > 0:
        x = Dropout(dropout)(x)
    x = Dense(4096, activation='relu', name='full7')(x)
    if dropout > 0:
        x = Dropout(dropout)(x)

    # Predictions Block
    x = Dense(8, activation='softmax', name='full8')(x)
    model = Model(inputs, x, name='cnns_model')

    return model
