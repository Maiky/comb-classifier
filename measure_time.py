import time
import combdetection.config as conf
if not conf.ANALYSE_PLOTS_SHOW:
    #needed to plot images on flip
    import matplotlib
    matplotlib.use('Agg')

import matplotlib.pyplot as plt
import sys
import numpy as np
from combdetection import util, keras_helpers, models
from scipy.misc import imread, imresize, imsave
import cv2
from combdetection.segmentation import get_superpixel_segmentation
from skimage.segmentation import mark_boundaries
import os.path
import  glob

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



def get_color_mapped_image(nn_output):

    comb_image = np.zeros((nn_output.shape[0], nn_output.shape[1], 3))
    image = np.zeros((nn_output.shape[0], nn_output.shape[1], 3))

    bee_image = np.zeros((nn_output.shape[0], nn_output.shape[1], 3))

    for i in range(nn_output.shape[0]):
        for j in range(nn_output.shape[1]):
            probs = nn_output[i][j]
            pk = np.argmax(probs)
            #print(probs[pk])
            #print((i,j))
            #print(pk)
            #print(conf.CLASS_COLOR_MAPPING.get(pk))
            value =  np.zeros((len(conf.CLASS_LABEL_MAPPING),3))
            value_comb = np.zeros((len(conf.CLASS_LABEL_MAPPING),3))
            value_bee = np.zeros((len(conf.CLASS_LABEL_MAPPING),3))
            for k,v in conf.CLASS_COLOR_MAPPING.items():
                prob = probs[k]
                if(prob > 0.1):
                    if("bee" in conf.CLASS_LABEL_MAPPING.get(k)):
                        value_bee[k]= np.asarray(v)*prob
                        value[k]= np.asarray(v)*prob
                    else:
                        value_comb[k]= np.asarray(v)*prob
                        value[k]= np.asarray(v)*prob
            comb_image[i,j] =np.sum(value_comb, axis=0)
            bee_image[i,j] =np.sum(value_bee, axis=0)
            image[i,j] =np.sum(value, axis=0)
            #conf.CLASS_COLOR_MAPPING.get(pk) #
            #print(image[i,j])
            #print((i,j))

    #print(image[157:177, 220:240])
    #print(image[177,240])
    #print(image[0,0])
    return image, comb_image, bee_image

if __name__ == '__main__':
    network_name = sys.argv[1]
    image_file = sys.argv[2]
    #dataset_file = sys.argv[3]

    config_file = conf.TRAINING_WEIGHTS_PATH+network_name+".hdf5"

    times = []
    files = []

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
    print('path is directory, scan for *.jpeg')
    current_dir = os.getcwd()
    os.chdir(image_file)
    for file in glob.glob("*.jpeg"):
        #image = imread(os.path.realpath(file))


        image, compressed_image, targetsize = util.compress_image_for_network(os.path.realpath(file))
        #superpixel_masks, segments = get_superpixel_segmentation(np.transpose([compressed_image]*3, axes=(1,2,0)), 25)

        start = time.time()
        saliency = f(
               compressed_image.reshape((1, 1, compressed_image.shape[0], compressed_image.shape[1])))
        end = time.time()
        print(end - start)
        times.append(end-start)

    print(times)
    print(np.sum(times)/len(times))
