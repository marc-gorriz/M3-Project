import keras.backend as K

from keras.layers import Dense, MaxPooling2D, Flatten, Input, Convolution2D, BatchNormalization, Activation, \
    GaussianNoise, Dropout
from keras.models import Model
from keras.regularizers import l2


def deep_model(img_width, img_height, regularization=0.1, batch_normalization=False, dropout=None, stddev=0):
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
        if dropout is not None:
            x = Dropout(dropout)(x)

        x = Dense(2048, activation='relu', W_regularizer=l2(regularization), name='fc2')(x)
        if dropout is not None:
            x = Dropout(dropout)(x)

        x = Dense(8, activation='softmax', name='predictions')(x)

        model = Model(inputs, x, name='deep_model')

        return model
