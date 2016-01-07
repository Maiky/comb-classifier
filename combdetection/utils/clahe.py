import cv2
import sys
import os.path
import glob

img = cv2.imread('Cam_1_20150901065804_457410.jpeg',0)


def createclahe():
    # create a CLAHE object (Arguments are optional).
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl1 = clahe.apply(img)
    cv2.imwrite('clahe_3.jpg',cl1)



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
        img = cv2.imread(os.path.realpath(file),0)
        # create a CLAHE object (Arguments are optional).
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cl1 = clahe.apply(img)
        print('generate '+'clahe_'+os.path.basename(file))
        cv2.imwrite('clahe_'+os.path.basename(file),cl1)