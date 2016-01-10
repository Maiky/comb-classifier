import sys
import combdetection.utils.generator as generator
import combdetection.config  as conf
import combdetection.small_conv
import numpy as np
import pickle
import os.path
from keras.utils import np_utils
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.objectives import mse
from combdetection import util, keras_helpers, models
import matplotlib.pyplot as plt
from scipy.misc import imread, imresize
from combdetection.segmentation import get_superpixel_segmentation
from skimage.segmentation import mark_boundaries

def get_saliency_image(Ysamples, targetsize):
    stride = 2
    image_saliency  = np.zeros((targetsize[0], targetsize[1], 3), dtype=np.float)

    cnt = 0
    y = 0
    while (y + conf.NETWORK_SAMPLE_SIZE[1] < targetsize[1]):
        x = 0
        while (x + conf.NETWORK_SAMPLE_SIZE[0] < targetsize[0]):
            cenx = int(x + conf.NETWORK_SAMPLE_SIZE[0] / 2)
            ceny = int(y + conf.NETWORK_SAMPLE_SIZE[1] / 2)
            image_saliency[cenx, ceny] = Ysamples[cenx][ceny]
            image_saliency[cenx+1, ceny+1] = Ysamples[cenx][ceny]
            cnt += 1
            x += stride
        y += stride

    return image_saliency


if __name__ == '__main__':
    network_name = sys.argv[1]
    image_file = sys.argv[2]
    #dataset_file = sys.argv[3]
    
    config_file = conf.TRAINING_WEIGHTS_PATH+network_name+".hdf5"

    image = imread(image_file)



    image, compressed_image, targetsize = util.compress_image_for_network(image_file)
    superpixel_masks, segments = get_superpixel_segmentation(np.transpose([compressed_image]*3, axes=(1,2,0)), 25)

    fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True)
    #ax = ax.flatten()
    ax.imshow(mark_boundaries(compressed_image, segments))
    ax.set_title('superpixel')
    plt.show()
    exit()
    #ax[3].imshow(seg_labeled_image * 255)
    #ax[3].set_title('labeled image')
    #ax[0].set_xticks([])
    #ax[0].set_yticks([])

    saliency_network = models.get_saliency_network(train=True)
    saliency_network.load_weights(config_file)

    saliency_conv_model = models.get_saliency_network(train=False)

    #Xsamples = util.get_sliding_window_samples(compressed_image, targetsize)


    #Ysamples = saliency_network.predict_proba(Xsamples)
    #Ysamples_classes = Ysamples[:,0:3]
    #print(Ysamples_classes.shape)
    #print(np.count_nonzero(Ysamples_classes[:,0]))
    #print(np.count_nonzero(Ysamples_classes[:,1]))
    #print(np.count_nonzero(Ysamples_classes[:,2]))




    f = keras_helpers.get_convolution_function(saliency_network, saliency_conv_model)
    saliency = f(
           compressed_image.reshape((1, 1, compressed_image.shape[0], compressed_image.shape[1])))
    im = np.transpose(saliency[0][0], axes=(1,2,0)).copy()
    im = imresize(im,(compressed_image.shape[0], compressed_image.shape[1]))
    superpixel_masks, segments = get_superpixel_segmentation(np.transpose([compressed_image]*3, axes=(1,2,0)), 1000)
    seg_labeled_image = np.zeros((im.shape[0], im.shape[1], im.shape[2]))
    labeled_image = np.zeros((im.shape[0], im.shape[1], im.shape[2]))

    for mask in superpixel_masks:
        class_mask = np.transpose([mask]*im.shape[2], axes=(1,2,0))
        sp = im * class_mask
        #print(np.count_nonzero(sp))
        cl_res = sp.reshape((sp.shape[0]*sp.shape[1], sp.shape[2]))
        classes = np.argmax(sp,axis=2)
        ct_classes = []
        for i in range(3):
            ct = np.sum(np.asarray((classes == i)).astype(int)*mask)
            #print("amount "+str(i)+":"+str(ct))
            ct_classes.append(ct)
        #print(ct_classes)
        class_label = np.argmax(ct_classes)
        #print(class_label)
        #mean = np.mean(classes)
        #print(mean)
        #class_label = int(np.round(mean))
        #print(class_label)

        color = conf.CLASS_COLOR_MAPPING.get(class_label)
        labeled_mask = class_mask * color
        #
        #ig, ax = plt.subplots(nrows=1, ncols=1)
        #ax[0].imshow(image, cmap=plt.cm.gray)
        #ax[0].setTitle("original")
        #ax.imshow(labeled_mask)
        #plt.show()
        seg_labeled_image = seg_labeled_image + labeled_mask

        labeled_image = labeled_image + sp
        #mean = np.mean(sp)
        #print(mean.shape)
        #class_label = np.argmax(mean)


    #im = np.reshape(im, (im.shape[1], im.shape[2], im.shape[0]))
    #Y_classes= np.asarray(im)
    #print(im.shape)
    #sal_image = get_saliency_image(im, targetsize)
    #norm =im * 255

    fig, ax = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True)
    ax = ax.flatten()
    ax[0].imshow(compressed_image, cmap=plt.cm.gray)
    ax[0].set_title('original image')
    #ax[0].setTitle("original")
    ax[1].imshow(labeled_image*255)
    ax[1].set_title('network output')
    #ax[2] = fig[2].add_subplot(1, 1, 1)
    ax[2].imshow(mark_boundaries(compressed_image, segments))
    ax[2].set_title('superpixel')
    ax[3].imshow(seg_labeled_image * 255)
    ax[3].set_title('labeled image')
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    if(conf.ANALYSE_PLOTS_SAVE):
        fn = conf.ANALYSE_PLOTS_PATH+network_name+"_labeled_image.png"
        plt.savefig(fn)
    if(conf.ANALYSE_PLOTS_SHOW):
        plt.axis('off')
        plt.show()
    #candidates = util.get_candidates(saliency, saliency_threshold)
    #rois, saliencies = util.extract_rois(candidates, saliency, image)