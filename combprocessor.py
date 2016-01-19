import combdetection.config as conf
#if not conf.GENERATOR_OUTPUT:
    #needed to plot images on flip
    # import matplotlib
    # matplotlib.use('Agg')

import matplotlib.pyplot as plt
import combdetection.utils.dbadapter as dba
from scipy.misc import imread, imresize
from  skimage.morphology import binary_closing, black_tophat,convex_hull_image
from skimage.filters import threshold_otsu, threshold_adaptive
from skimage.filters.rank import median
from skimage.morphology import disk
from skimage import measure
import cv2
import argparse
import numpy as np
import datetime


# parser = argparse.ArgumentParser(description='')
# parser.add_argument('integers', metavar='i', type=int, nargs='+',
#                    help='an integer for the accumulator')
# parser.add_argument('--sum', dest='accumulate', action='store_const',
#                    const=sum, default=max,
#                    help='sum the integers (default: find the max)')
#
# args = parser.parse_args()
# print(args.accumulate(args.integers))


img = imread(conf.ANALYSE_PLOTS_PATH+"nn_ds_bees_noclahe_c4_bees.png")
combs = img[:,:,2]
honey = img[:,:,0]
sm_f = median(honey, disk(5))
global_thresh = threshold_otsu(sm_f)
print(global_thresh)
global_thresh = 150
binary_global= np.asarray(sm_f > global_thresh).astype('uint8')

binary_global *= 255

from skimage import measure
#closed = binary_closing(binary_global)

#bees = black_tophat(img_add,selem =np.ones((40,40)))#binary_closing(binary_global)#binary_closing(honey)#black_tophat(img)
fig, ax = plt.subplots(nrows=2, ncols=2)

ax = ax.flatten()
ax[0].imshow(img)
ax[1].imshow(honey)
ax[2].imshow(sm_f)
ax[3].imshow(cv2.cvtColor(binary_global, cv2.COLOR_GRAY2RGB))
plt.show()

#print(np.max(binary_global))
#cv2.find_contours(binary_global,0.8, fully_connected='high')
#contours = cv2.findContours(binary_global, mode=cv2.RETR_TREE)
#cv2.findContours()
#blubb = cv2.cvtColor(binary_global, cv2.COLOR_GRAY2RGB)
#blubb = cv2.cvtColor(blubb, cv2.COLOR_RGB2GRAY)
im2, contours, hierarchy = cv2.findContours(binary_global.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#print(hierarchy)
im_c = cv2.cvtColor(binary_global, cv2.COLOR_GRAY2RGB)

hierarchy = hierarchy[0] # get the actual inner list of hierarchy descriptions



polygons = {}

# For each contour, find the bounding rectangle and draw it
for component in zip(contours, hierarchy, range(len(hierarchy))):
    currentContour = component[0]
    currentHierarchy = component[1]
    index = component[2]
    x,y,w,h = cv2.boundingRect(currentContour)
    if currentHierarchy[3] < 0:
        # these are the outermost parent components
        cv2.drawContours(im_c, contours,index ,color=(0,255,0), thickness=3)
        if(polygons.get(index) is not None):
            tmp = polygons.get(index)
            tmp.append(currentContour)
            polygons[index] = tmp
        else:
            polygons[index] = [currentContour]

        #cv2.rectangle(im_c,(x,y),(x+w,y+h),(0,255,0),3)
    else:
        # these are the innermost child components
        #cv2.rectangle(im_c,(x,y),(x+w,y+h),(0,0,255),3)
        cv2.drawContours(im_c, contours, index,color=(0,0,255), thickness=3)
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



#db = dba.Adapter()
#db.insert_comb_layout(polygons, 1, datetime.date(2105, 12, 12), 1)


#)
#im_c = cv2.drawContours(im_c, contours,-1,(0,0,255),3)
fig2, ax2 = plt.subplots()
ax2.imshow(im_c, interpolation='nearest') #cmap=plt.cm.gray

#for n, contour in enumerate(contours):
#    ax2.plot(contour[:, 1], contour[:, 0], linewidth=2)

ax2.axis('image')
ax2.set_xticks([])
ax2.set_yticks([])
plt.show()
