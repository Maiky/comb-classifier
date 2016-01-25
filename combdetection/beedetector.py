import combdetection.models_new as md
import combdetection.config as conf
import combdetection.keras_helpers as kh
from skimage.filters import threshold_otsu, threshold_adaptive
from skimage.filters.rank import median
from skimage.morphology import disk
import numpy as np
from skimage.morphology import binary_erosion, binary_dilation
import matplotlib.pyplot as plt
import cv2
import combdetection.utils.dbadapter as dba
from scipy.stats import threshold
from scipy.misc import imread, imresize, imsave
from fractions import Fraction
from combdetection.segmentation import get_superpixel_segmentation
from skimage.segmentation import mark_boundaries
from skimage.feature import peak_local_max
from skimage.filters.rank import median


class BeeDetector():
    def __init__(self):
        self.index_mapping = {}

        i = 1
        print(conf.CLASS_LABEL_MAPPING.items())
        for k, v in conf.CLASS_LABEL_MAPPING.items():
            if "bee" in v:
                self.index_mapping[i] = k
                i += 1

    def _extract_comb_classes(self, image):

        selected_classes = list(self.index_mapping.values())

        # get bee-classes
        # class_probs = np.transpose(image[0][0], axes=(1,2,0)).copy()
        masks = image[:, :, selected_classes]

        # generate a column with zeros with index[0] to identifiy unlabeled pixels in arg_max
        new_column = np.zeros((masks.shape[0], masks.shape[1], 1))
        shifted_mask = np.concatenate((new_column, masks), axis=2)


        sum_mask = np.zeros((masks.shape[0], masks.shape[1]))
        if len(selected_classes) > 0:
            trans = np.transpose(masks, axes=(2,0,1))
            for i in range(trans.shape[0]):
                mask = trans[i]
                sum_mask = sum_mask + mask
        else:
            sum_mask = masks

        return shifted_mask, sum_mask

    def _gen_absolute_classes(self, mask):
        print("hallo")

    def detect_bees_per_image(self, classified_image, orig_image, datetime, camera_id, targetsize, storeInDb = True):
        shifted_masks, mask = self._extract_comb_classes(classified_image)

        bees = mask/np.max(mask) #mask[:, :, 1]

        bee_thres = threshold(bees, threshmin=0.5, threshmax=1, newval=0)


        bee_thres = cv2.resize(bee_thres, (targetsize[1], targetsize[0]))

        binary = np.asarray(bee_thres > 0).astype('uint8')

        # im2, contours, hierarchy = cv2.findContours(binary.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

        valid_contours = self._extract_valid_contours(binary, bee_thres)
        # for cnt in contours:
        #     area = cv2.contourArea(cnt)
        #     if area  > 200:
        #
        #         print('too_big')
        #     elif area >15:
        #         valid_contours.append(cnt)

        # hierarchy = hierarchy[0] # get the actual inner list of hierarchy descriptions
        # polygons = {}

        # Iterate over contours and hierarchie to find outer and inner shapes
        # for component in zip(contours, hierarchy, range(len(hierarchy))):
        #     currentContour = component[0]
        #     currentHierarchy = component[1]
        #     index = component[2]
        #     x,y,w,h = cv2.boundingRect(currentContour)
        #     if currentHierarchy[3] < 0:
        #         # these are the outermost parent components
        #         if(polygons.get(index) is not None):
        #             tmp = polygons.get(index)
        #             tmp.append(currentContour)
        #             polygons[index] = tmp
        #         else:
        #             polygons[index] = [currentContour]
        #
        #         #cv2.rectangle(im_c,(x,y),(x+w,y+h),(0,255,0),3)
        #     else:
        #         # these are the innermost child components
        #         p_index = index
        #
        #         #find highest parent
        #         while hierarchy[p_index, 3] >= 0:
        #             p_index = hierarchy[p_index, 3]
        #
        #         if(polygons.get(p_index) is not None):
        #             tmp = polygons.get(p_index)
        #             tmp.append(currentContour)
        #             polygons[p_index] = tmp
        #         else:
        #             polygons[p_index] = [currentContour]
        # return polygons

        # area = cv2.contourArea(cnt)
        im_c = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
        conts = cv2.drawContours(im_c, valid_contours, -1, (0, 255, 0), 2)
        fig, ax = plt.subplots(nrows=2, ncols=1)
        ax = ax.flatten()
        global_thres = threshold_otsu(bee_thres)
        print(global_thres)
        tmp0 = ax[0].imshow(bees, interpolation='nearest')
        fig.colorbar(tmp0)
        tm1 = ax[1].imshow(bee_thres, interpolation='nearest')
        # ax[2].imshow(orig_image, cmap=plt.cm.gray)
        # ax[3].imshow(conts)
        plt.show()
        fig, ax = plt.subplots(nrows=2, ncols=1)
        tmp0 = ax[0].imshow(orig_image, interpolation='nearest', cmap=plt.cm.gray)
        tm1 = ax[1].imshow(conts, interpolation='nearest')
        plt.show()
        fig, ax = plt.subplots(nrows=2, ncols=1)
        im_c = cv2.cvtColor(orig_image, cv2.COLOR_GRAY2RGB)
        im_c = imresize(im_c, (binary.shape[0], binary.shape[1]))
        conts = cv2.drawContours(im_c, valid_contours, -1, (0, 255, 0), 10)
        ax[0].imshow(conts, interpolation='nearest')

        im = conts.copy()

        for cnt in valid_contours:
            #ellipse = cv2.fitEllipse(cnt)
            #im = cv2.ellipse(im,ellipse,(0,0,255),10)
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            im = cv2.drawContours(im, [box], 0, (255, 0, 0), 10)

            if storeInDb:
                db = dba.Adapter()
                db.insert_bee(datetime, camera_id, cnt, 0, 1, True)

            # cv2.ellipse(conts)
            # rows,cols = im.shape[:2]
            # [vx,vy,x,y] = cv2.fitLine(cnt, cv2.DIST_L2,0,0.01,0.01)
            # lefty = int((-x*vy/vx) + y)
            # righty = int(((cols-x)*vy/vx)+y)
            # img = cv2.line(im,(cols-1,righty),(0,lefty),(0,0,255),2)
        ax[1].imshow(im, interpolation='nearest')

        # tm1 = ax[1].imshow(conts, interpolation='nearest')
        plt.show()

    def _extract_valid_contours(self, image, orig_image, offset_x=None, offset_y=None, offset_sub_x=None,
                                offset_sub_y=None, i=1):
        valid_contours = []
        im2, contours, hierarchy = cv2.findContours(image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        offset = [0, 0]
        if offset_sub_x is not None:
            offset = np.add([offset_sub_x, 0], offset)
        if offset_sub_y is not None:
            offset = np.add([0, offset_sub_y], offset)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            point, size, angle = cv2.minAreaRect(cnt)
            width = np.max(size)
            height = np.min(size)
            if area > conf.BEEDETECTOR_MAX_BEE_AREA or width > conf.BEEDETECTOR_MAX_WIDTH or height > conf.BEEDETECTOR_MAX_HEIGHT:
                x, y, w, h = cv2.boundingRect(cnt)
                # conts = cv2.drawContours(cv2.cvtColor(image*255, cv2.COLOR_GRAY2RGB), cnt, -1, (0,255,0), 2)
                orig_sub_image = orig_image[y:y + h, x:x + w]
                # ids =list(np.ndindex(orig_sub_image.shape))
                mask = np.zeros((orig_sub_image.shape[0], orig_sub_image.shape[1]))
                cnt_tmp = cnt.reshape(cnt.shape[0], cnt.shape[2])
                cnt_tmp = np.subtract(cnt_tmp, [x, y])
                cv2.fillPoly(mask, pts=[cnt_tmp], color=(255, 255, 255))
                mask = np.asarray(mask > 0).astype('uint8')
                orig_sub_image = np.multiply(orig_sub_image, mask)
                # mask = mask
                # result = cv2.pointPolygonTest(cnt,i, False)
                # if result:
                #    mask[i[1], i[0]] = 1
                # mask = np.asarray(mask).reshape(orig_sub_image.shape)
                # print(mask.shape)

                # orig_sub_image = orig_image[y:y+h, x:x+w], disk(1)
                thres_min = 0.2 + sum(Fraction(1, d ** 2) for d in range(2, i + 3))
                print(i)
                print(thres_min)
                # print(0.1**i)
                bee_thres = threshold(orig_sub_image, threshmin=thres_min, threshmax=1,
                                      newval=0)  # threshold_adaptive(orig_sub_image, 10)#
                sub_image = np.asarray(bee_thres > 0).astype('uint8')
                sub_image =np.asarray(binary_dilation(bee_thres)).astype('uint8')#image[y:y+h, x:x+w] #np.asarray(binary_erosion(,disk(1))).astype('uint8')
                fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=True)
                ax = ax.flatten()
                tmp0 = ax[0].imshow(orig_sub_image, interpolation='nearest') #image[y:y+h,x:x+w]
                tm1 = ax[1].imshow(bee_thres, interpolation='nearest')
                ax[0].set_xticks([])
                ax[0].set_yticks([])
                #ax[1].set_title(str(thres_min))
                plt.show()
                sub_x = x
                sub_y = y
                if offset_sub_x is not None:
                    sub_x += offset_sub_x
                if offset_sub_y is not None:
                    sub_y += offset_sub_y
                sm_contours = self._extract_valid_contours(sub_image, orig_sub_image, x, y, sub_x, sub_y, i + 1)
                valid_contours.extend(sm_contours)
            elif area > conf.BEEDETECTOR_MIN_BEE_AREA:
                # print(rect)
                cnt[:, 0] = np.add(cnt[:, 0], offset)
                valid_contours.append(cnt)
        return valid_contours

    def detect_peaks(self, image):
        import numpy as np
        from scipy.ndimage.filters import maximum_filter
        from scipy.ndimage.morphology import generate_binary_structure, binary_erosion
        import matplotlib.pyplot as pp

        """
        Takes an image and detect the peaks usingthe local maximum filter.
        Returns a boolean mask of the peaks (i.e. 1 when
        the pixel's value is the neighborhood maximum, 0 otherwise)
        """

        # define an 8-connected neighborhood
        neighborhood = generate_binary_structure(8, 8)

        # apply the local maximum filter; all pixel of maximal value
        # in their neighborhood are set to 1
        local_max = maximum_filter(image, footprint=neighborhood) == image
        # local_max is a mask that contains the peaks we are
        # looking for, but also the background.
        # In order to isolate the peaks we must remove the background from the mask.

        # we create the mask of the background
        background = (image == 0)

        # a little technicality: we must erode the background in order to
        # successfully subtract it form local_max, otherwise a line will
        # appear along the background border (artifact of the local maximum filter)
        eroded_background = binary_erosion(background, structure=neighborhood, border_value=1)

        # we obtain the final mask, containing only peaks,
        # by removing the background from the local_max mask
        detected_peaks = local_max - eroded_background

        return detected_peaks
