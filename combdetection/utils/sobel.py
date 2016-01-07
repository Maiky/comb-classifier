import cv2
import os.path
import sys
import glob




def createSobel(file):
    img = cv2.imread(file,cv2.CV_8UC1)
    image_base, extention= os.path.splitext(os.path.basename(file))
    #img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # create a CLAHE object (Arguments are optional).
    grad_x = cv2.Sobel(img, cv2.CV_16S,1,0,ksize=5)
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    grad_y  = cv2.Sobel(img,cv2.CV_16S,0,1,ksize=5)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    sobel = cv2.addWeighted(abs_grad_x, .5, abs_grad_y, 0.5, 0)
    th3 = cv2.adaptiveThreshold(sobel,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    cv2.namedWindow("image", cv2.WINDOW_AUTOSIZE)
    cv2.imshow("image",th3)
    cv2.waitKey(0)
    #cv2.imwrite('sobel_'+image_base+extention,sobel)


if __name__ == '__main__':

    filename = sys.argv[1]
    if not os.path.isfile(filename):
        if not os.path.isdir(filename):
            print('file not found')
            exit()
        else:
            files = []
            print('path is directory, scan for *.jpeg')
            os.chdir(filename)
            for file in glob.glob("*.jpeg"):
                files.append(os.path.abspath(file))
    else:
        files = [filename]
    for file in files:
        createSobel(os.path.realpath(file))