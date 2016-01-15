
import sys
import os.path
import glob
from skimage.exposure import  equalize_adapthist
from scipy.misc import imsave, imread
from skimage.util import img_as_float
import numpy as np



if __name__ == '__main__':

    image_path = sys.argv[1]
    #target = sys.argv[1]

    #gen = generator.Generator(target)
    #gen.generate_dataset(directory)

    if not os.path.isfile(image_path):
        if not os.path.isdir(image_path):
            raise IOError('file not exists')
        else:
            files = []
            print('path is directory, scan for *.jpeg')
            os.chdir(image_path)
            for file in glob.glob("*.jpeg"):
                files.append(os.path.abspath(file))
    else:
        files = [image_path]

    for file in files:
        print('process '+os.path.realpath(file))
        img = imread(os.path.realpath(file),flatten=True)
        #img = img_as_float(img/255)
        #print(img.shape)
        #print(np.max(img))
        cl1 = equalize_adapthist(img/255)
        print('generate '+'clahe_'+os.path.basename(file))
        imsave('clahe_'+os.path.basename(file),cl1)
        #cv2.imwrite('clahe_'+os.path.basename(file),cl1)