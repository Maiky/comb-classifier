
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


class CombClassifier:

    def __init__(self):

        #create convolution-function to apply convolution-network to whole image

        # cnn = md.get_comb_net(train=True) # .get_saliency_network(train=True)
        # cnn.load_weights(conf.NETWORK_WEIGHTS_PATH+conf.NETWORK_COMBS_WEIGHTS_FILE_NAME+'.hdf5')
        #
        # conv_model = md.get_comb_net(train=False)
        #
        # f = kh.get_convolution_function(cnn, conv_model)
        # #saliency = f(
        #  #  compressed_image.reshape((1, 1, compressed_image.shape[0], compressed_image.shape[1])))
        # self.cnn = f

        self.scene_masks = []
        self.index_mapping = {}

        i = 1
        for k,v in conf.NETWORK_COMBS_CLASS_LABEL_MAPPING.items():
            if "bee" not in v:
                self.index_mapping[i] = k
                i += 1

    def _extract_comb_classes(self, image):


        selected_classes = list(self.index_mapping.values())

        # get comb-classes
        #class_probs = np.transpose(image[0][0], axes=(1,2,0)).copy()
        mask = image[:,:, selected_classes]

        # generate a column with zeros with index[0] to identifiy unlabeled pixels in arg_max
        new_column = np.zeros((mask.shape[0], mask.shape[1], 1))
        shifted_mask = np.concatenate((new_column, mask),axis=2)
        self.scene_masks.append(shifted_mask)



    def _getAverageMask(self):

        res = np.zeros((self.scene_masks[0].shape[0], self.scene_masks[0].shape[1], self.scene_masks[0].shape[2]))
        count = 1/len(self.scene_masks)
        for v in self.scene_masks:
            print(v.shape)
            res = res+(count*v)
        return res


    def getMasksForClasses(self, resultmask):

        max_mat = np.argmax(resultmask, axis=2)

        masks = {}
        for k,v in self.index_mapping.items():
            mask = np.asarray((max_mat == k)).astype('uint8')
            masks[v] = mask
        return masks


    def gen_comb_state_per_day(self, classified_images, date, camera_id, targetsize, storeInDb = True):
        masks = self._getClasses(classified_images)
        mask_polygons ={}
        for k, mask in masks.items():
            scaled_mask = cv2.resize( mask, (targetsize[1], targetsize[0]))
            polygons = self._get_polygons_for_mask(scaled_mask)
            if storeInDb:
                self._store_polygons_in_db(polygons, k, camera_id, date)
            mask_polygons[k] = polygons
        return masks,mask_polygons


    def _getClasses(self, images):
        print('I#m in getClasses')


        for image in images:
            print("start processing image..")
            self._extract_comb_classes(image)

        #fig, ax = plt.subplots(nrows=2, ncols=3, sharey=True, sharex=True)
        #ax = ax.flatten()

        #for i, mask in zip(range(len(self.scene_masks)),self.scene_masks):
        #    ax[i].imshow(mask, interpolation='nearest')
        #    #ax[i].set_title('Einzelbild')
        #ax[0].set_xticks([])
        #ax[0].set_yticks([])
        #plt.show()
        #exit()

        av_mask = self._getAverageMask()
        im_av_mask = av_mask
        im_av_mask[:,:,2] = 0
        plt.imshow(im_av_mask)
        plt.show()
        exit()
        #ax[len(self.scene_masks)].set_title('Mittelwert')

        masks = self.getMasksForClasses(av_mask)
        #for mask in masks.values():

        im_mask = np.zeros((3,masks[1].shape[0], masks[1].shape[1]))
        im_mask[1,:] = masks[1]
        #im_mask[2,:] = masks[2]
        im_mask = np.transpose(im_mask, axes=(1,2,0))
        plt.imshow(im_mask)
        plt.show()
        exit()


        for k, mask in masks.items():
            masks[k] = self._postprocessMask(mask)



        return masks

    def _get_polygons_for_mask(self, mask):
        im2, contours, hierarchy = cv2.findContours(mask.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        hierarchy = hierarchy[0] # get the actual inner list of hierarchy descriptions
        polygons = {}

        # Iterate over contours and hierarchie to find outer and inner shapes
        for component in zip(contours, hierarchy, range(len(hierarchy))):
            currentContour = component[0]
            currentHierarchy = component[1]
            index = component[2]
            x,y,w,h = cv2.boundingRect(currentContour)
            if currentHierarchy[3] < 0:
                # these are the outermost parent components
                if(polygons.get(index) is not None):
                    tmp = polygons.get(index)
                    tmp.append(currentContour)
                    polygons[index] = tmp
                else:
                    polygons[index] = [currentContour]

                #cv2.rectangle(im_c,(x,y),(x+w,y+h),(0,255,0),3)
            else:
                # these are the innermost child components
                p_index = index

                #find highest parent
                while hierarchy[p_index, 3] >= 0:
                    p_index = hierarchy[p_index, 3]

                if(polygons.get(p_index) is not None):
                    tmp = polygons.get(p_index)
                    tmp.append(currentContour)
                    polygons[p_index] = tmp
                else:
                    polygons[p_index] = [currentContour]
        return polygons

    def _store_polygons_in_db(self, polygons, class_id, camera_id, date):
        db = dba.Adapter()
        db.insert_comb_layout(polygons, camera_id, date, class_id)



    def _postprocessMask(self, mask):

        #mask *= 255
        smoothed = median(mask, disk(5))
        #global_thresh = threshold_otsu(smoothed)
        #global_thresh = global_thresh
        #binary= np.asarray(smoothed > global_thresh).astype('uint8')
        #block_size = 40
        #binary_adaptive = threshold_adaptive(smoothed, block_size, offset=10)
        #binary*= 255
        return smoothed
