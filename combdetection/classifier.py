

import combdetection.models_new as md
import combdetection.config_new as conf
import combdetection.keras_helpers as kh
from skimage.filters import threshold_otsu, threshold_adaptive
from skimage.filters.rank import median
from skimage.morphology import disk
import numpy as np
import matplotlib.pyplot as plt
import cv2
import combdetection.utils.dbadapter as dba
from keras.utils import generic_utils


class Classifier():


    def __init__(self):

        #create convolution-function to apply convolution-network to whole image

        cnn = md.get_comb_net(train=True) # .get_saliency_network(train=True)
        cnn.load_weights(conf.NETWORK_WEIGHTS_PATH+conf.NETWORK_COMBS_WEIGHTS_FILE_NAME+'.hdf5')

        conv_model = md.get_comb_net(train=False)

        f = kh.get_convolution_function(cnn, conv_model)
        #saliency = f(
         #  compressed_image.reshape((1, 1, compressed_image.shape[0], compressed_image.shape[1])))
        self.cnn = f


    def _classifyImage(self, image):
        output = self.cnn(image.reshape(1,1, image.shape[0], image.shape[1]))
        class_probs = np.transpose(output[0][0], axes=(1,2,0)).copy()
        return class_probs

    def classifyImages(self, images):
        ret = []
        progbar = generic_utils.Progbar(len(images))
        for image in images:
            class_prob = self._classifyImage(image)
            ret.append(class_prob)
            progbar.add(1)
        return ret
