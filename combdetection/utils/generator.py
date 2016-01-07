import os
import h5py
import combdetection.config
from skimage import data, measure
import numpy as np
from skimage import measure
from matplotlib import pyplot as plt
import matplotlib.path as mplPath
#import matplotlib.transforms as trans
from numpy import random
import sys
import os, glob
from distutils.util import strtobool
from sklearn.cross_validation import train_test_split


class Generator(object):
    def __init__(self,dataset_path,append=True):
        self.dataset_path = dataset_path
        if(os.path.exists(self.dataset_path)):
            if(append):
                self.f = h5py.File(self.dataset_path, 'r+')
            else:
                os.remove(self.dataset_path)
                self.f = h5py.File(self.dataset_path, "w")
        else:
             self.f = h5py.File(self.dataset_path, "w")



    def load_traindata(self, test_size=0.3, equal_class_sizes=True, dataset="", max_size_per_set=None):
        if len(dataset) > 0:
            if (not dataset in self.f):
                raise Exception("data-set not found in file")
            else:
                datasets= [dataset]
        else:
            datasets = []
            for name in self.f:
                datasets.append(name)

        X = []
        y = []

        for set in datasets:
            print("load samples from dataset "+set)
            dset = self.f.get(set)
            X_tmp, y_tmp = self._add_samples(dset, equal_class_sizes, max_size_per_set=max_size_per_set)
            X.extend(X_tmp)
            y.extend(y_tmp)
        print(" got total"+str(np.shape(X))+" samples..")
        print(" generate training/test-data with test-size:"+str(test_size))
        X_train, X_test, y_train, y_test= train_test_split(X,y, test_size=test_size)
        return X_train, X_test, y_train, y_test


    def _add_samples(self, dset, equal_class_sizes=True, max_size_per_set=None):
        """
       loads a training/test-set generated with generator

        @type  dset: h5py.Dataset
        @param dset: current dataset
        @rtype:   np.array, np.array
        @return: true, if generation successful, false otherwise
        """
        max_sample_size = None
        if(max_size_per_set is not None):
            max_sample_size = int(np.round(max_size_per_set / len(combdetection.config.NETWORK_CLASS_LABELS)))
        elif(combdetection.config.NETWORK_TRAIN_MAX_SAMPLES is not None):
            max_sample_size = int(np.round(combdetection.config.NETWORK_TRAIN_MAX_SAMPLES / len(combdetection.config.NETWORK_CLASS_LABELS)))

        min_sample_count = -1
        if(equal_class_sizes):
            for name in dset:
                ds = dset.get(name)
                if name in combdetection.config.NETWORK_CLASS_LABELS:
                    if(min_sample_count == -1) | (ds.len() < min_sample_count):
                        min_sample_count = ds.len()
        if(max_sample_size is not None):
            if (max_sample_size < min_sample_count):
                min_sample_count = max_sample_size
        X = []
        y = []
        #iterate over classes
        for name in dset:
            if name in combdetection.config.NETWORK_CLASS_LABELS:
                ds = dset.get(name)
                print('     add training-data for label: '+ name)

                values = ds[()]
                if(equal_class_sizes):
                    if(min_sample_count>0):
                        values = values[0:(min_sample_count)]
                    else:
                        values = []
                    sample_count = min_sample_count
                else:
                    sample_count = ds.len()
                X.extend(values)
                print(str(np.shape(X)))
                encoded_class = combdetection.config.NETWORK_CLASS_LABELS.get(name)
                y.extend([encoded_class] * sample_count)
                print('     added '+str(sample_count)+" samples")

        return X,y






    def show_details(self,dataset=""):
        """
       loads a training/test-set generated with generator

        @type  dataset: str
        @param dataset: name of the data-set, if not specified, load all data-sets in file
        @rtype:   bool
        @return: true, if generation successful, false otherwise
        """
        if len(dataset) > 0:
            if (not dataset in self.f):
                raise Exception("data-set not found in file")
            else:
                datasets= [dataset]
        else:
            datasets = []
            for name in self.f:
                datasets.append(name)

        for set in datasets:
            self._print_details_for_dataset(set)


    def _print_details_for_dataset(self, dataset):
        print("sample-count details for "+dataset+":")
        for name in self.f.get(dataset):
            dset = self.f.get(dataset).get(name)

            print("    "+name+": "+str(dset.len()))



    def generate_dataset(self, image_path, mask_type="",compression=combdetection.config.GENERATOR_COMPRESSION_FAKTOR):

        #check if path is directory or file,
        # parse directory if necessary
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

        #check mask_types
        if len(mask_type) > 0:
            if mask_type in combdetection.config.NETWORK_CLASS_LABELS.keys():
                mask_types = [mask_type]
            else:
                raise Exception("mask-type "+mask_type+" not found")
        else:
            mask_types = combdetection.config.NETWORK_CLASS_LABELS.keys();

        #iterate over all found files
        for file in files:
            image_base, extention= os.path.splitext(os.path.basename(file))
            dataset_group = image_base
            if dataset_group in self.f:
                answer= self._user_yes_no_query("image-samples for "+dataset_group+" already exits. should image be skipped?")
                print(answer)
                if answer == 1 :
                    continue

            grp = self.f.require_group(dataset_group)
            for mask in mask_types:
                self._generate_samples_for_mask(grp, os.path.realpath(file), mask, False, compression)


    def _generate_samples_for_mask(self,grp, file, mask_type, accept_outside=False, compression=None):
        import cv2
        """
        Return the x intercept of the line M{y=m*x+b}.  The X{x intercept}
        of a line is the point at which it crosses the x axis (M{y=0}).

        This function can be used in conjuction with L{z_transform} to
        find an arbitrary function's zeros.

        @type  grp: h5py.Group
        @param grp: group in the dataset-file
        @type  b: str
        @param b: absolute path to the original image
        @type mask_type: str
        @rtype:   number
        @return:  the x intercept of the line M{y=m*x+b}.
        """
        name= grp.name
        if(mask_type in grp):
            print("nothing to do in "+name+"/"+mask_type+", type exists")
            return
        print("start processing "+ name)
        output = combdetection.config.GENERATOR_OUTPUT

        samples = []

        bees_mask = ("bees" in mask_type)
        size = combdetection.config.GENERATOR_SAMPLE_SIZE
        if(compression is not None):
            compressed_size = [int(np.round(size[0]/compression)), int(np.round(size[1]/compression))]
        else:
            compressed_size = size
        vector_size = compressed_size[0]* compressed_size[1]
        max_shift = combdetection.config.GENERATOR_MAX_SHIFT
        print("generate samples for mask-type:"+mask_type+ " with size:"+str(size)+" max-shift:"+str(max_shift)+"and compression:"+str(compression)+".")
        print("new size of samples:"+str(compressed_size)+" and in vector:"+str(vector_size))

        image_base, extention= os.path.splitext(os.path.basename(file))
        image_path,img = os.path.split(os.path.abspath(file))
        #target_directory = image_path+"/"+image_base+"_"+mask_type

        #print(name)
        #print(image_path+"/"+image_base+'_'+mask_type+'.png')
        #exit()
        r = data.imread(image_path+"/"+image_base+'_'+mask_type+'.png', True)
        orig = cv2.imread(file, cv2.CV_8U)
        #if not target == False:
            #target_directory = target_directory+"/"+mask_type

        #if not os.path.exists(target_directory):
         #   print("generate target folder "+target_directory)
         #   os.makedirs(target_directory)

        #print(image.shape)
        #print(image)
        #subimage =image[0:80,0:80]
        #x, y = np.ogrid[-np.pi:np.pi:100j, -np.pi:np.pi:100j]
        #r = np.sin(np.exp((np.sin(x)**3 + np.cos(y)**2)))
        contours = measure.find_contours(r, 0)
        print("found "+ str(len(contours))+" polygons")
        # Display the image and plot all contours found
        if output:
            fig, ax = plt.subplots(nrows=1, ncols=1)
            fig1, ax1 = plt.subplots(nrows=1, ncols=1)
            fig2, ax2 = plt.subplots(nrows=5, ncols=5, sharex=True, sharey=True,)
            ax2 = ax2.flatten()
            ax.imshow(r, interpolation='nearest', cmap=plt.cm.gray)
            ax1.imshow(orig, interpolation='nearest', cmap=plt.cm.gray)

        all_sample_count = 0
        # plt.savefig('./figures/mnist_miscl.png', dpi=300)

        for n,contour in enumerate(contours):
            print('process contour '+str(n)+'...')
            if output:
                ax1.plot(contour[:, 1], contour[:, 0], linewidth=2)
                #ax[1].plot(contour[:, 1], contour[:, 0], linewidth=2)

            bbBox = contour#np.transpose([contour[:,0], contour[:,1]])
            #if output:
                #ax[0].plot(bbBox[:, 1], bbBox[:, 0], linewidth=2)
            bbPath = mplPath.Path(bbBox, closed=True)

            #get outer bounding box
            max_x = np.max(bbBox[:,0], axis=0)
            max_y = np.amax(bbBox[:,1], axis=0)

            min_x = np.amin(bbBox[:,0], axis=0)
            min_y = np.amin(bbBox[:,1], axis=0)

            #if output:
                #ax[0].plot([min_y, max_y, max_y, min_y, min_y], [min_x,min_x,max_x,max_x, min_x],linewidth=2)

            #if(min_x +size[0] > max_x):
            #    samples_count = 1
            #    single_sample = True
            #else:
            #    samples_count = np.round(((max_x-min_x))*((max_y-min_y))*ratio, 0)
            #    single_sample = False


            single_sample_col = False
            sample_start_x = min_x
            samples_in_cols = 0
            if accept_outside & (sample_start_x > max_x-size[0]):
                single_sample_col = True
                offset_x = np.round((sample_start_x - max_x-size[0])/2)
                sample_start_x -= offset_x
            while sample_start_x <= (max_x-size[0]):
                single_sample_row = False
                positions_y =[pos[1] for pos in contour if pos[0] == sample_start_x]
                y_min_tmp = np.amin(positions_y, axis=0)
                y_max_tmp = np.amax(positions_y, axis=0)
                sample_start_y = y_min_tmp
                if (y_max_tmp- size[1] < y_min_tmp) & accept_outside:
                    single_sample_row = True
                    offset_y = np.round((sample_start_y - y_max_tmp-size[1])/2)
                    sample_start_y -= offset_y
                samples_in_row = 0
                while (sample_start_y <= (y_max_tmp-size[0])) | single_sample_row:
                    #bbPath = mplPath.Path([[min_x,min_y],[min_x,max_y],[max_x,max_y],[max_x,min_y],[min_x,min_y]], closed=True)
                    x_1 = [sample_start_x, sample_start_y]
                    x_2 = [sample_start_x, sample_start_y+size[1]]
                    y_2 = [sample_start_x+size[0], sample_start_y]
                    y_1 = [sample_start_x+size[0], sample_start_y+size[1]]
                    rect =[x_1,x_2,y_1,y_2]

                    #sampleBox = trans.Bbox.from_bounds(sample_start_x[i], sample_start_y[i],80,80)
                    #if i< 10:
                        #print(rect)
                    res =bbPath.contains_points(rect)

                    bbPath.unit_rectangle()
                    #if i <10:
                        #print(res)

                    #if bbPath.intersects_bbox(sampleBox, filled=True):

                    #if sample is fully in bound box add
                    if res.all() | bees_mask:
                        samples_in_row += 1
                        subimage = orig[sample_start_x:(sample_start_x+size[0]),sample_start_y:(sample_start_y+size[1])]
                        if(compression is not None):
                            subimage = cv2.resize(subimage,(compressed_size[0], compressed_size[1]))
                        if(all_sample_count< 25) & output:
                            ax2[all_sample_count].imshow(subimage,cmap=plt.cm.gray, interpolation='nearest')
                            ax2[all_sample_count].set_title('%d_%d' % (sample_start_x,sample_start_y ))
                        image_data = subimage.reshape((vector_size))
                        samples.append(image_data)
                            #plt.imsave(target_directory+"/"+image_base+"_sample_"+str(sample_start_x)+"_"+str(sample_start_y)+"_"+mask_type+".jpg",subimage,cmap=plt.cm.gray)
                        all_sample_count += 1
                        if output:
                            ax.plot([x_1[1], x_2[1], y_1[1], y_2[1], x_1[1]],[x_1[0],x_2[0],y_1[0],y_2[0], x_1[0]],  linewidth=2)
                    #else:
                        #print('no match')
                        #ax.plot( [x_1[1], x_2[1], y_1[1], y_2[1], x_1[1]],[x_1[0],x_2[0],y_1[0],y_2[0], x_1[0]],'-.',    linewidth=2)
                    if single_sample_row:
                        break
                    else:
                        sample_start_y += np.random.randint(1, max_shift)
                samples_in_cols += samples_in_row
                sys.stderr.write('\rSample %d:, total in contour: %d' % (n, samples_in_cols))
                sys.stderr.flush()
                sample_start_x += np.random.randint(1, max_shift)
                samples_in_row = 0
                if(single_sample_col):
                    break
            print('done')

        if output:
            #ax2[0].set_xticks([])
            #ax2[0].sets_yticks([])
            #plt.tight_layout()
            plt.show()
            print("generated "+ str(all_sample_count)+" samples for type "+mask_type)
            print(str(np.shape(samples)))
        dset = grp.create_dataset(mask_type,data=samples)
        #dset.attrs['class_label'] = mask_type
        print("")




    def _user_yes_no_query(self,question):
        sys.stdout.write('%s [y/n]\n' % question)
        while True:
            try:
                return strtobool(input().lower())
            except ValueError:
                sys.stdout.write('Please respond with \'y\' or \'n\'.\n')