from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import sys
import combdetection.utils.generator, combdetection.config
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from keras.utils import np_utils
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.objectives import mse

def plot_decision_regions(X, y, classifier, resolution=0.02):

    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                         np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=cmap(idx),
                    marker=markers[idx], label=cl)

if __name__ == '__main__':
    dataset_file = sys.argv[1]
    config_file = ""
    if(len(sys.argv) >=3):
        config_file = sys.argv[2]


    sc = StandardScaler()
    gen = combdetection.utils.generator.Generator(dataset_file)
    X_train, X_test, y_train, y_test= gen.load_traindata()
    X_train = np.asarray(X_train)
    X_test = np.asarray(X_test)
    y_test = np.asarray(y_test)
    y_train = np.asarray(y_train)

    #X_train = X_train.reshape(X_train.shape[0], 1, combdetection.config.NETWORK_SAMPLE_SIZE[0], combdetection.config.NETWORK_SAMPLE_SIZE[1])
    #X_test = X_test.reshape(X_test.shape[0], 1, combdetection.config.NETWORK_SAMPLE_SIZE[0], combdetection.config.NETWORK_SAMPLE_SIZE[1])
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    X_train_std = sc.fit_transform(X_train)
    X_test_std = sc.fit_transform(X_test)

    pca = PCA(n_components=9)

    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.fit_transform(X_test)

    #batch_size = 128
    nb_classes = 3
    nb_epoch = 15

    size = X_train_pca.shape
    print(size)
    # input image dimensions
    img_rows, img_cols = 3,3
    # number of convolutional filters to use
    nb_filters = 32
    # size of pooling area for max pooling
    nb_pool = 2
    # convolution kernel size
    nb_conv = 3

    X_train = X_train_pca.reshape(X_train_pca.shape[0], 1, img_cols, img_rows)
    X_test = X_test_pca.reshape(X_test_pca.shape[0], 1,  img_cols, img_rows)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    #X_train /= 255
    #X_test /= 255
    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    print('Rows: %d, columns: %d' % (X_train.shape[0], X_train.shape[1]))
    print('Rows: %d, columns: %d' % (X_test.shape[0], X_test.shape[1]))


    model = Sequential()

    model.add(Convolution2D(nb_filters, nb_conv, nb_conv,
                            border_mode='valid',
                            input_shape=(1, img_rows, img_cols)))
    model.add(Activation('relu'))
    model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adadelta')

    model.fit(X_train, Y_train,nb_epoch=nb_epoch,
              show_accuracy=True, verbose=1, validation_data=(X_test, Y_test))

    score = model.evaluate(X_test, Y_test, show_accuracy=True, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

    plot_decision_regions(X_train_pca, y_train, classifier=model)
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.legend(loc='lower left')
    plt.tight_layout()
    # plt.savefig('./figures/pca3.png', dpi=300)
    plt.show()
    #
    # colors = ['r', 'b', 'g']
    # markers = ['s', 'x', 'o']
    # inv_class_mapping = {v: k for k, v in combdetection.config.NETWORK_CLASS_LABELS.items()}
    #
    # for l, c, m in zip(np.unique(y_train), colors, markers):
    #     plt.scatter(X_train_pca[y_train==l, 0],
    #                 X_train_pca[y_train==l, 1],
    #                 c=c, label=inv_class_mapping.get(l), marker=m)
    #
    # plt.xlabel('PC 1')
    # plt.ylabel('PC 2')
    # plt.legend(loc='lower left')
    # plt.tight_layout()
    # # plt.savefig('./figures/pca2.png', dpi=300)
    # plt.show()
    #
    # exit()
    # print(pca.explained_variance_ratio_.shape)
    # print(pca.explained_variance_ratio_[0:20].shape)
    # plt.bar(range(1, 21), pca.explained_variance_ratio_[0:20], alpha=0.5, align='center')
    # plt.step(range(1, 21), np.cumsum(pca.explained_variance_ratio_[0:20]), where='mid')
    # plt.ylabel('Explained variance ratio')
    # plt.xlabel('Principal components')
    # plt.show()