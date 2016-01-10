# import the necessary packages
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from scipy.misc import imread, imresize
import matplotlib.pyplot as plt
import argparse
import pystruct.models as psm
import numpy as np
#import combdetection.config



def get_crf(original_image):
    psm.GraphCRF()

def get_superpixel_segmentation(image, num_segments = 10):
    # apply SLIC and extract (approximately) the supplied number
    # of segments
    #image = img_as_float(image)
    segments = slic(image, n_segments = num_segments)
    superpixel_masks = []
    for i in range(np.max(segments)+1):
        mask = np.asarray((segments == i)).astype(int)
        superpixel_masks.append(mask)
    return superpixel_masks, segments
