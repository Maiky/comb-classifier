#!/usr/bin/python3

import json
from os.path import isfile

import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import generic_utils
from keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
from keras.layers.core import Flatten
from sklearn.metrics import auc, label_ranking_average_precision_score,accuracy_score, confusion_matrix,coverage_error, label_ranking_loss
import theano
import combdetection.config as conf
import combdetection.utils.generator as gen
import os
import h5py
from keras.utils import np_utils
import matplotlib.pyplot as plt






class LoggingCallback(Callback):
    def __init__(self, network_name):
        super().__init__()
        self.network_name = network_name

    def load_log(self):
        fp = conf.TRAINING_LOG_PATH+self.network_name+".hd5f"
        if (os.path.exists(fp)):
            f = h5py.File(fp, "r")
            bds = f.get("batch_hist")
            eds = f.get("epoch_hist")
            self.batch_hist = bds[()]
            self.epoch_hist = eds[()]
            return True
        return False

    def on_train_begin(self, logs={}):
        self.batch_hist = []
        self.epoch_hist = []

    def on_batch_end(self, batch, logs={}):
        self.batch_hist.append((logs.get('loss'), logs.get('acc')))

    def on_epoch_end(self, batch, logs={}):
        self.epoch_hist.append((logs.get('val_loss'), logs.get('val_acc')))

    def on_train_end(self,logs={}):
        if(conf.TRAINING_LOG_SAVE):
            fp = conf.TRAINING_LOG_PATH+self.network_name+".hd5f"
            if (os.path.exists(fp)):
                os.remove(fp)
            f = h5py.File(fp, "w")
            f.create_dataset("batch_hist", data=self.batch_hist)
            f.create_dataset("epoch_hist", data=self.epoch_hist)

"""
 helper-class to train the network
"""
class Trainer():

    def _preprocess_training_data(self, dataset_name):

        #load generator-class and get training samples
        g = gen.Generator(dataset_name)
        g.show_details()
        X_train, X_test, y_train, y_test= g.load_traindata()
        X_train = np.asarray(X_train)
        X_test = np.asarray(X_test)
        y_test = np.asarray(y_test)
        y_train = np.asarray(y_train)

        X_train = X_train.reshape((X_train.shape[0], 1, conf.NETWORK_SAMPLE_SIZE[0], conf.NETWORK_SAMPLE_SIZE[1]))
        X_test = X_test.reshape((X_test.shape[0], 1, conf.NETWORK_SAMPLE_SIZE[0], conf.NETWORK_SAMPLE_SIZE[1]))
        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')
        X_train /= 255
        X_test /= 255
        print('X_train shape:', X_train.shape)
        print(X_train.shape[0
              ], 'train samples')
        print(X_test.shape[0], 'test samples')

        nb_classes = np.max(y_train) +1

        # convert class vectors to binary class matrices
        Y_train = np_utils.to_categorical(y_train,nb_classes)
        Y_test = np_utils.to_categorical(y_test,nb_classes)

        return X_train,X_test,Y_train,Y_test


    def fit(self, model, dataset_name, network_filename,
            nb_epoch=100, batch_size=128):

        X_train, X_test, Y_train, Y_test = self._preprocess_training_data(dataset_name)
        #callback to save weights after each epoch
        checkpointer = ModelCheckpoint(filepath=conf.TRAINING_WEIGHTS_PATH+network_filename+".hdf5", verbose=0, save_best_only=True)

        #callback to stop training if monitor-param stays the same for patinence-count epochs
        stopper = EarlyStopping(monitor='val_loss', patience=3, verbose=1)

        #custom callback to store accuracy
        log = LoggingCallback(network_filename)

        callbacks = [checkpointer, stopper, log]

        #or callback in callbacks:
            #callback._set_model(model)
            #callback.on_train_begin()

        model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,callbacks=callbacks,
                  show_accuracy=True, verbose=1, validation_data=(X_test, Y_test))


        return log


    def show_training_performance(self, log, save_plots = True):
        """
        @param log: LoggingCallback
        @return:
        """

        batches = np.array([log[0] for log in log.batch_hist])
        plt.plot(range(len(batches)), batches, color="blue")
        plt.ylabel("loss")
        plt.xlabel("batches")
        if(conf.ANALYSE_PLOTS_SAVE):
            plt.savefig(conf.ANALYSE_PLOTS_PATH+log.network_name+"_train_batches.png")
        if(conf.ANALYSE_PLOTS_SHOW):
            plt.show()


        epochs = np.array([log[0] for log in log.epoch_hist])
        plt.plot(range(len(epochs)), epochs, color="red")
        plt.ylabel("val_loss")
        plt.xlabel("epochs")
        if(conf.ANALYSE_PLOTS_SAVE):
            plt.savefig(conf.ANALYSE_PLOTS_PATH+log.network_name+"_train_epochs.png")
        if(conf.ANALYSE_PLOTS_SHOW):
            plt.show()



    def show_prediciton_performance(self,y_true, y_pred, y_score):
        acc_score = accuracy_score(y_true, y_pred)
        conf_mat =confusion_matrix(y_true, y_pred)
        cov_err = coverage_error(y_true, y_score)
        lraps = label_ranking_average_precision_score(y_true,y_score)
        lrl = label_ranking_loss(y_true, y_score)
