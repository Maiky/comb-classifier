#!/usr/bin/python3

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.objectives import mse
import combdetection.config as conf

def get_comb_net(train=False):

    nb_classes = len(conf.CLASS_LABEL_MAPPING)

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

    model.add(Convolution2D(nb_classes,11,11))
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