
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

    image = imread(image_file)


    image, compressed_image, targetsize = util.compress_image_for_network(image_file)
    #superpixel_masks, segments = get_superpixel_segmentation(np.transpose([compressed_image]*3, axes=(1,2,0)), 25)


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
    im, comb_image, bee_image = get_color_mapped_image(im)
    comb_im = imresize(comb_image,(compressed_image.shape[0], compressed_image.shape[1]))
    bee_im = imresize(bee_image,(compressed_image.shape[0], compressed_image.shape[1]))
    im = imresize(im,(compressed_image.shape[0], compressed_image.shape[1]))

    #  superpixel_masks, segments = get_superpixel_segmentation(np.transpose([compressed_image]*3, axes=(1,2,0)), 1000)
    # seg_labeled_image = np.zeros((im.shape[0], im.shape[1], im.shape[2]))
    # labeled_image = np.zeros((im.shape[0], im.shape[1], im.shape[2]))
    #
    # for mask in superpixel_masks:
    #     class_mask = np.transpose([mask]*im.shape[2], axes=(1,2,0))
    #     sp = im * class_mask
    #     #print(np.count_nonzero(sp))
    #     cl_res = sp.reshape((sp.shape[0]*sp.shape[1], sp.shape[2]))
    #     classes = np.argmax(sp,axis=2)
    #     ct_classes = []
    #     for i in range(3):
    #         ct = np.sum(np.asarray((classes == i)).astype(int)*mask)
    #         #print("amount "+str(i)+":"+str(ct))
    #         ct_classes.append(ct)
    #     #print(ct_classes)
    #     class_label = np.argmax(ct_classes)
    #     #print(class_label)
    #     #mean = np.mean(classes)
    #     #print(mean)
    #     #class_label = int(np.round(mean))
    #     #print(class_label)
    #
    #     color = conf.CLASS_COLOR_MAPPING.get(class_label)
    #     labeled_mask = class_mask * color
    #     #
    #     #ig, ax = plt.subplots(nrows=1, ncols=1)
    #     #ax[0].imshow(image, cmap=plt.cm.gray)
    #     #ax[0].setTitle("original")
    #     #ax.imshow(labeled_mask)
    #     #plt.show()
    #     seg_labeled_image = seg_labeled_image + labeled_mask
    #
    #     labeled_image = labeled_image + sp
    #     #mean = np.mean(sp)
    #     #print(mean.shape)
    #     #class_label = np.argmax(mean)
    #
    #
    # #im = np.reshape(im, (im.shape[1], im.shape[2], im.shape[0]))
    # #Y_classes= np.asarray(im)
    # #print(im.shape)
    # #sal_image = get_saliency_image(im, targetsize)
    # #norm =im * 255

    fig, ax = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True)
    fig.patch.set_visible(False)
    ax = ax.flatten()
    ax[0].imshow(compressed_image, cmap=plt.cm.gray)
    ax[0].set_title('original image')
    #ax[0].setTitle("original")
    ax[1].imshow(im)
    imsave( conf.ANALYSE_PLOTS_PATH+network_name+"_masks.png", im)
    ax[1].set_title('network output')
    ax[2].imshow(comb_im)
    imsave( conf.ANALYSE_PLOTS_PATH+network_name+"_combs.png", comb_im)
    ax[2].set_title('combs')
    ax[3].imshow(bee_im)
    imsave( conf.ANALYSE_PLOTS_PATH+network_name+"_bees.png", bee_im)
    ax[3].set_title('bees')
    #ax[2] = fig[2].add_subplot(1, 1, 1)
    #ax[2].imshow(mark_boundaries(compressed_image, segments))
    #ax[2].set_title('superpixel')
    #ax[3].imshow(seg_labeled_image * 255)
    #ax[3].set_title('labeled image')
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    plt.tight_layout()
    if(conf.ANALYSE_PLOTS_SAVE):
        fn = conf.ANALYSE_PLOTS_PATH+network_name+"_labeled_image.png"
        plt.savefig(fn)
    if(conf.ANALYSE_PLOTS_SHOW):
        plt.axis('off')
        plt.show()
    print("done")
    #candidates = util.get_candidates(saliency, saliency_threshold)
    #rois, saliencies = util.extract_rois(candidates, saliency, image)