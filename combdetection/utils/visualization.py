
import numpy as np
import combdetection.config_new as conf

def get_colored_image(class_matrix):
    image = np.zeros((class_matrix.shape[0],class_matrix.shape[1],3))
    for i in range(np.max(class_matrix)):
        mask = np.asarray(class_matrix == i).astype('uint8')
        color_mask = np.tile(conf.NETWORK_COMBS_CLASS_COLOR_MAPPING.get(i), (mask.shape[0],mask.shape[1]))
        image += color_mask* mask
    return image

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