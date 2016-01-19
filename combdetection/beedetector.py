

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
from scipy.stats import threshold


class BeeDetector():


    def __init__(self):

        self.index_mapping = {}

        i = 1
        for k,v in conf.NETWORK_COMBS_CLASS_LABEL_MAPPING.items():
            if "bee" in v:
                self.index_mapping[i] = k
                i += 1

    def _extract_comb_classes(self, image):

        selected_classes = list(self.index_mapping.values())

        # get bee-classes
        #class_probs = np.transpose(image[0][0], axes=(1,2,0)).copy()
        mask = image[:,:, selected_classes]

        # generate a column with zeros with index[0] to identifiy unlabeled pixels in arg_max
        new_column = np.zeros((mask.shape[0], mask.shape[1], 1))
        shifted_mask = np.concatenate((new_column, mask),axis=2)
        return shifted_mask


    def _gen_absolute_classes(self,mask):
        print("hallo")

    def detect_bees_per_image(self,classified_image, orig_image,  datetime, camera_id):
        mask = self._extract_comb_classes(classified_image)

        bees = mask[:,:,1]
        bee_thres = threshold(bees, threshmin=0.7, threshmax=1, newval=0)

        fig, ax = plt.subplots(nrows=2, ncols=2)
        ax = ax.flatten()
        tmp0 = ax[0].imshow(bees, interpolation='nearest')
        tm1 = ax[1].imshow(bee_thres, interpolation='nearest')
        ax[2].imshow(orig_image, cmap=plt.cm.gray)

        fig.colorbar(tmp0)
        plt.show()


