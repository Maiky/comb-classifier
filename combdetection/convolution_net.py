
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.objectives import mse
from combdetection import config


def get_filter_network():
    model = Sequential()

    model.add(Convolution2D(16, 2, 2,
                            input_shape=(1, config.NETWORK_SAMPLE_SIZE[0], config.NETWORK_SAMPLE_SIZE[1]),
                            border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(16, 2, 2, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(16, 2, 2))
    model.add(Activation('relu'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.2))

    model.add(Convolution2D(32, 2, 2, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(32, 2, 2, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(32, 2, 2))
    model.add(Activation('relu'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.2))

    model.add(Convolution2D(64, 2, 2, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 2, 2, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 2, 2))
    model.add(Activation('relu'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.2))

    model.add(Flatten())

    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    #model.compile(loss="mse", optimizer="adam")
    model.compile(loss='categorical_crossentropy', optimizer='adadelta')

    return model
