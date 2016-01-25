

'''
This sample demonstrates SEEDS Superpixels segmentation
Use [space] to toggle output mode

Usage:
  seeds.py [<video source>]

'''

import numpy as np
import cv2
import matplotlib.pyplot as plt
import combdetection.util as util



# built-in module
import sys
from combdetection.segmentation import get_superpixel_segmentation
from skimage.segmentation import mark_boundaries
from scipy.misc import imread, imresize, imsave

if __name__ == '__main__':

    try:
        fn = sys.argv[1]
    except:
        fn = 0

    def nothing(*arg):
        pass

    image, compressed_image, targetsize = util.compress_image_for_network(fn)
    superpixel_masks, segments = get_superpixel_segmentation(np.transpose([compressed_image]*3, axes=(1,2,0)), 500)
    plt.imshow(mark_boundaries(compressed_image, segments))
    plt.show()

    #
    # cv2.namedWindow('SEEDS',cv2.WINDOW_AUTOSIZE)
    # cv2.createTrackbar('Number of Superpixels', 'SEEDS', 400, 1000, nothing)
    # cv2.createTrackbar('Iterations', 'SEEDS', 4, 12, nothing)
    #
    # seeds = None
    # display_mode = 0
    # num_superpixels = 400
    # prior = 2
    # num_levels = 4
    # num_histogram_bins = 5
    #
    #
    # while True:
    #     img = cv2.imread(fn)
    #     img = cv2.resize(img, (1000,750))
    #     converted_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #     height,width,channels = converted_img.shape
    #     num_superpixels_new = cv2.getTrackbarPos('Number of Superpixels', 'SEEDS')
    #     num_iterations = cv2.getTrackbarPos('Iterations', 'SEEDS')
    #
    #     if not seeds or num_superpixels_new != num_superpixels:
    #         num_superpixels = num_superpixels_new
    #         seeds = cv2.ximgproc.createSuperpixelSEEDS(width, height, channels,
    #                 num_superpixels, num_levels, prior, num_histogram_bins)
    #         color_img = np.zeros((height,width,3), np.uint8)
    #         color_img[:] = (0, 255, 255)
    #     seeds.iterate(converted_img, num_iterations)
    #
    #     # retrieve the segmentation result
    #     labels = seeds.getLabels()
    #
    #
    #
    #     # labels output: use the last x bits to determine the color
    #     num_label_bits = 2
    #     labels &= (1<<num_label_bits)-1
    #     labels *= 1<<(16-num_label_bits)
    #
    #
    #     mask = seeds.getLabelContourMask(False)
    #
    #     # stitch foreground & background together
    #     mask_inv = cv2.bitwise_not(mask)
    #     result_bg = cv2.bitwise_and(img, img, mask=mask_inv)
    #     result_fg = cv2.bitwise_and(color_img, color_img, mask=mask)
    #     result = cv2.add(result_bg, result_fg)
    #
    #     if display_mode == 0:
    #         cv2.imshow('SEEDS', result)
    #     elif display_mode == 1:
    #         cv2.imshow('SEEDS', mask)
    #     else:
    #         cv2.imshow('SEEDS', labels)
    #
    #     ch = cv2.waitKey(1)
    #     if ch == 27:
    #         break
    #     elif ch & 0xff == ord(' '):
    #         display_mode = (display_mode + 1) % 3
    # cv2.destroyAllWindows()