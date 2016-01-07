
import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imread, imresize

def generate_labels(mask):
    mask_mapped = np.copy(mask)
    labels = np.zeros((mask.shape[0], mask.shape[1], 6), dtype=np.uint8)
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            # = find_nearest(mask[i,j])
            mask_mapped[i,j],labels[i,j] = find_nearest(mask[i,j])
    return mask_mapped, labels

def label_generate_mask(labels):
    mask = np.zeros((labels.shape[0], labels.shape[1], 3), dtype=np.uint8)
    color_mapping_arr = np.asarray([v for k,v in color_mapping.items()])
    print(color_mapping_arr.shape)
    print(np.transpose(color_mapping_arr))
    for i in range(labels.shape[0]):
        for j in range(labels.shape[1]):
            mask[i,j] = np.dot(np.transpose(color_mapping_arr),labels[i,j])

    return mask



def find_nearest(value, encode_class=False):
    color_mapping_arr = [v for k,v in color_mapping.items()]
    array = np.asarray(color_mapping_arr)
    value = np.asarray(value)
    distances =[np.linalg.norm(v-value) for v in array]
    idx = np.asarray(distances).argmin()

    v = np.zeros(6)
    v[idx] = 1
    return color_mapping_arr[idx], v



color_mapping = {1:[255,255,0], 2: [100,100,100], 3:[100,100,0],  4: [255, 0, 0], 5:[0, 255, 0],  6: [0, 0, 255]}
class_mapping = {0:"bee_head", 1: "bee_breast", 2: "bee_back", 3: "tag", 4:"honey", 5:"comb"}

fn = sys.argv[1]




mask = imread(fn)
print(mask)
#mask = np.zeros(((6*20), 20, 3), dtype=np.uint8)
#mask[0:20] = color_mapping.get(1)
#mask[20:40] = color_mapping.get(2)
#mask[40:60] = color_mapping.get(3)
#mask[60:80] = color_mapping.get(4)
#mask[80:100] = color_mapping.get(5)
#mask[100:120] = color_mapping.get(6)


#print(mask)
mask_conv = cv2.cvtColor(mask, cv2.COLOR_RGB2BGR)
cv2.namedWindow("orig", cv2.WINDOW_FREERATIO)
cv2.imshow("orig", mask_conv)
cv2.waitKey(0)
masked_mapped, labels = generate_labels(mask)
print(masked_mapped)

cv2.namedWindow("mapped mask", cv2.WINDOW_FREERATIO)
cv2.imshow("mapped mask", cv2.cvtColor(masked_mapped, cv2.COLOR_RGB2BGR))
cv2.waitKey(0)
remapped_mask = label_generate_mask(labels)
cv2.namedWindow("remapped mask", cv2.WINDOW_FREERATIO)
cv2.imshow("remapped mask", cv2.cvtColor(remapped_mask, cv2.COLOR_RGB2BGR))
cv2.waitKey(0)
cv2.destroyAllWindows()


#print(np.unique(mask, True, False, True))
exit()