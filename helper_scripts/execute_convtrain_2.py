import sys
import combdetection.utils.generator as generator
import combdetection.config
import combdetection.small_conv
import combdetection.models
import numpy as np
import pickle
import os.path
from keras.utils import np_utils
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.objectives import mse


if __name__ == '__main__':
    dataset_file = sys.argv[1]
    config_file = ""
    if(len(sys.argv) >=3):
        config_file = sys.argv[2]

    # input image dimensions


    batch_size = 128
    nb_classes = 3
    nb_epoch = 15

    # input image dimensions
    img_rows, img_cols = 28, 28
    # number of convolutional filters to use
    nb_filters = 32
    # size of pooling area for max pooling
    nb_pool = 2
    # convolution kernel size
    nb_conv = 3
    # (X_train, y_train), (X_test, y_test) = mnist.load_data()
    # X_train = X_train[0:600]
    # y_train = y_train[0:600]
    #
    # X_test = X_test[0:400]
    # y_test = y_test[0:400]
    #
    # X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    # X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    # X_train = X_train.astype('float32')
    # X_test = X_test.astype('float32')
    # X_train /= 255
    # X_test /= 255
    # print('X_train shape:', X_train.shape)
    # print(X_train.shape[0], 'train samples')
    # print(X_test.shape[0], 'test samples')
    # # convert class vectors to binary class matrices
    # Y_train = np_utils.to_categorical(y_train, nb_classes)
    # Y_test = np_utils.to_categorical(y_test, nb_classes)

    gen = generator.Generator(dataset_file)
    X_train, X_test, y_train, y_test= gen.load_traindata()
    X_train = np.asarray(X_train)
    X_test = np.asarray(X_test)
    y_test = np.asarray(y_test)
    y_train = np.asarray(y_train)

    X_train = X_train.reshape(X_train.shape[0], 1, combdetection.config.NETWORK_SAMPLE_SIZE[0], combdetection.config.NETWORK_SAMPLE_SIZE[1])
    X_test = X_test.reshape(X_test.shape[0], 1, combdetection.config.NETWORK_SAMPLE_SIZE[0], combdetection.config.NETWORK_SAMPLE_SIZE[1])
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    print('Rows: %d, columns: %d' % (X_train.shape[0], X_train.shape[1]))
    print('Rows: %d, columns: %d' % (X_test.shape[0], X_test.shape[1]))
    # nn = combdetection.small_conv.get_filter_network()
    # if (len(config_file) == 0) | (not os.path.exists(config_file)):
    #     history = nn.fit(X_train, y_train,nb_epoch=nb_epoch,batch_size=batch_size, verbose=1, show_accuracy=True)
    #     nn.save_weights(config_file)
    #     #pickle.dump(nn, open(config_file, "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    # else:
    #     nn.load_weights(config_file)
    #     #nn = pickle.load(open(config_file, "rb"))
    # score = nn.evaluate(X_test, Y_test, show_accuracy=True, verbose=0)
    # print('Test score:', score[0])
    # print('Test accuracy:', score[1])


    model = combdetection.models.get_saliency_network(train=True)

    model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
              show_accuracy=True, verbose=1, validation_data=(X_test, Y_test))
    model.save_weights(config_file, overwrite=True)
    score = model.evaluate(X_test, Y_test, show_accuracy=True, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])