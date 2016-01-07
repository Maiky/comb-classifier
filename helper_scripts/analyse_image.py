import sys
import combdetection.utils.generator as generator
import combdetection.config
import combdetection.small_conv
import numpy as np
import pickle
import cv2
import os.path
from keras.utils import np_utils
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.objectives import mse
from combdetection import util, keras_helpers, models
import matplotlib.pyplot as plt

def get_saliency_image(Ysamples, targetsize):
    stride = 5
    image_saliency  = np.zeros((targetsize[0], targetsize[1], 3), dtype=np.float)

    cnt = 0
    y = 0
    while (y + combdetection.config.NETWORK_SAMPLE_SIZE[1] < targetsize[1]):
        x = 0
        while (x + combdetection.config.NETWORK_SAMPLE_SIZE[0] < targetsize[0]):
            cenx = int(x + combdetection.config.NETWORK_SAMPLE_SIZE[0] / 2)
            ceny = int(y + combdetection.config.NETWORK_SAMPLE_SIZE[1] / 2)
            image_saliency[cenx, ceny] = Ysamples[cnt][0:3]
            cnt += 1
            x += stride
        y += stride

    return image_saliency

if __name__ == '__main__':
    config_file = sys.argv[1]
    image_file = sys.argv[2]
    #dataset_file = sys.argv[3]
    image = cv2.imread(image_file)
    image, compressed_image, targetsize = util.compress_image_for_network(image_file)
    saliency_network = models.get_saliency_network()
    saliency_network.load_weights(config_file)

    saliency_conv_model = models.get_saliency_network(train=False)

    #Xsamples = util.get_sliding_window_samples(compressed_image, targetsize)


    #Ysamples = saliency_network.predict_proba(Xsamples)
    #Ysamples_classes = Ysamples[:,0:3]
    #print(Ysamples_classes.shape)
    #print(np.count_nonzero(Ysamples_classes[:,0]))
    #print(np.count_nonzero(Ysamples_classes[:,1]))
    #print(np.count_nonzero(Ysamples_classes[:,2]))


    #sal_image = get_saliency_image(Ysamples_classes, targetsize)
    #norm =cv2.normalize(sal_image,0,255)
    #cv2.namedWindow("saliency")
    #cv2.imshow("saliency", norm)
    #cv2.waitKey(0)
    #cv2.namedWindow("original")
    #cv2.imshow("original", image)
    #cv2.waitKey(0)

    f = keras_helpers.get_convolution_function(saliency_network, saliency_conv_model)
    saliency = f(
           compressed_image.reshape((1, 1, compressed_image.shape[0], compressed_image.shape[1])))
    print(saliency)
    #candidates = util.get_candidates(saliency, saliency_threshold)
    #rois, saliencies = util.extract_rois(candidates, saliency, image)