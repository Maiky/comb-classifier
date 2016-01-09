#!/usr/bin/python3

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.objectives import mse
import combdetection.config


def get_multi_scale_net(f1, f2):

    model = Sequential()

    #filter layer siz filter size 7 and tanh as activation-layer
    model.add(Convolution2D(f1, 7,7, activation='tanh'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Convolution2D(f2,1,1))


    # model.add(Convolution2D(32, 3, 3, input_shape=(1, None, None),
    #                         activation='relu', border_mode='same'))
    # model.add(Dropout(0.2))
    # model.add(Convolution2D(128, 16, 16, activation='tanh'))
    # model.add(Dropout(0.5))
    # model.add(Convolution2D(1, 1, 1, activation='sigmoid'))
    #
    # if train:
    #     model.add(Flatten())
    #
    # model.compile('adam', mse)
    #
    # return model

