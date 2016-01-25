#!/usr/bin/python3

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.objectives import mse
import combdetection.config as conf

def get_saliency_network_old(train=True):
    model = Sequential()
    model.add(Convolution2D(32, 3, 3, input_shape=(1, None, None),
                            activation='relu', border_mode='same'))
    model.add(Dropout(0.2))
    model.add(Convolution2D(128, 16, 16, activation='tanh'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(1, 1, 1, activation='sigmoid'))

    if train:
        model.add(Flatten())

    model.compile('adam', mse)

    return model

def get_saliency_network(train=False):

    batch_size = 128
    nb_classes = len(conf.CLASS_LABEL_MAPPING)
    nb_epoch = 15

    # input image dimensions
    img_rows, img_cols = conf.NETWORK_SAMPLE_SIZE[0], conf.NETWORK_SAMPLE_SIZE[1]
    # number of convolutional filters to use
    nb_filters = 32
    # size of pooling area for max pooling
    nb_pool = 2
    # convolution kernel size
    nb_conv = 3
    model = Sequential()

    model.add(Convolution2D(nb_filters, nb_conv, nb_conv,
                            border_mode='same',
                            input_shape=(1, img_cols, img_rows)))
    model.add(Activation('relu'))
    model.add(Convolution2D(nb_filters, nb_conv, nb_conv, border_mode='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    model.add(Dropout(0.25))

    # (45,45), (22,22) (11,11), (7,7)
    model.add(Convolution2D(nb_classes,7,7))
    if train:
        model.add(Flatten())
    #model.add(Dense(128))
    #model.add(Activation('relu'))
    #model.add(Dropout(0.5))
    #model.add(Dense(nb_classes))
    #don't use softmax
    model.add(Activation('sigmoid'))

    #model.compile(loss='categorical_crossentropy', optimizer='adadelta')
    model.compile('adam', mse)
    return model