
import argparse
import combdetection.combclassifier as cocl
import combdetection.beedetector as bd
import combdetection.classifier as cl
import combdetection.config_new as conf
import combdetection.utils.imageentry as ime
import os.path
import combdetection.util as util
import glob
from scipy.misc import imread, imresize, imsave
import re
import datetime




class HiveWatcher():

    def __init__(self):
        self.classifier = cl.Classifier()

    def _process_images_for_day(self, images,entries,  date, camera_id):

        """

        :param images:
        :param entries: combdetection.utils.imageentry.ImageEntry[]
        :param date:
        :param camera_id:
        :return:
        """

        print('start classifiying day '+str(date)+' for camera '+camera_id)

        classified_images = self.classifier.classifyImages(images)

        print('start bee-detection')
        beedtc = bd.BeeDetector()


        for entry, image, orig_image in zip(entries, classified_images, images):
            dt = datetime.datetime(entry.year, entry.month, entry.days, entry.hours, entry.minutes, entry.seconds, entry.microseconds)
            beedtc.detect_bees_per_image(image,orig_image, dt, camera_id, entry.origsize)

        #print("start comb-detection")
        #cc = cocl.CombClassifier()
        #masks, mask_polygons = cc.gen_comb_state_per_day(classified_images, date, camera_id, entries[0].origsize)

    def _load_images(self, files):

        cameras = {}
        #image_base, extention =
        #image_path, img = os.path.split(os.path.abspath(file))
        for file in files:
            basename, extension = os.path.splitext(os.path.basename(file))
            #m = re.search('Cam_([0-9])_([0-9]{4})([0-9]{2})([0-9]{2})([0-9]{2})([0-9]{2})([0-9]{2})_([0-9]*)', 'abcdef')
            m = re.split('Cam_(\d)_(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})_(\d*)', basename)
            camera_id = m[1]
            year = m[2]
            month = m[3]
            days = m[4]
            hours= m[5]
            minutes= m[6]
            seconds =m[7]
            microseconds = m[8]

            #image, compressed_image, targetsize = util.compress_image_for_network(file)

            if cameras.get(camera_id) is not None:
                day = cameras.get(camera_id)
            else:
                day = {}
            datestr= str(year)+"-"+str(month)+"-"+str(days)
            if day.get(datestr) is not None:
                images = day.get(datestr)
            else:
                images = []

            entry = ime.ImageEntry(int(camera_id), int(year), int(month), int(days), int(hours), int(minutes), int(seconds), int(microseconds),file)
            images.append(entry)
            day[datestr] = images
            cameras[camera_id] = day
        return cameras

    def process_files(self, files):
        cameras = self._load_images(files)

        for k, camera in cameras.items():
            for date, sequences in camera.items():
                images = []
                entries = []
                for entry in sequences:
                    #entry = sequences[i]
                    print("add image")
                    fn =entry.filename
                    image, compressed_image, targetsize = util.compress_image_for_network(fn)
                    entry.origsize = image.shape
                    images.append(compressed_image)
                    entries.append(entry)
                self._process_images_for_day(images,entries,  date, k)


                #if(len(sequences) > conf.HIVE_WATCHER_COMB_SAMPLES):






        






if __name__ == '__main__':
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser(description="main entrypoint for hivewatcher")
    ap.add_argument("-i", "--image", nargs='?',type=str, help = "Path to the image")
    ap.add_argument("-p", "--path", nargs='?', help = "Path to the images")
    ap.add_argument("-e", "--extension", nargs='?', help = "extension of images in path")
    ap.add_argument("-combs", "-combwatching", nargs='?', help="add combwatching")
    ap.add_argument("-bees", "--beewatching", nargs='?', help="add beewatching")
    args = vars(ap.parse_args())
    if not (args['image'] or args['path']):
        ap.error('No image(s) requested, add -i or -p')
        exit()

    if args['path'] and not args['extension']:
        ap.error('extension for images needed, if path is  given')
        exit()

    if args['image']:
        image_path = args['image']
        if not os.path.isfile(image_path):
            ap.error('image is no valid file')
            exit()
        else:
            files = [image_path]
    elif args['path']:
        path = args['path']
        if not os.path.isdir(path):
            ap.error('path does not exists')
            exit()
        else:
            files = []
            print('path is directory, scan for *.'+args['extension'])
            current_dir = os.getcwd()
            os.chdir(path)
            for file in glob.glob("*."+args['extension']):
                files.append(os.path.abspath(file))
            os.chdir(current_dir)

    hw = HiveWatcher()
    hw.process_files(files)

