import os
import h5py
from skimage import data, measure
import numpy as np
from skimage import measure
import matplotlib.path as mplPath
# import matplotlib.transforms as trans
from numpy import random
import sys
import os, glob
import combdetection.config as conf
from distutils.util import strtobool
from sklearn.cross_validation import train_test_split
from scipy.misc import imread, imresize
from scipy import stats

if not conf.GENERATOR_OUTPUT:
    #needed to plot images on flip
    import matplotlib
    matplotlib.use('Agg')

import matplotlib.pyplot as plt

"""
class to organize training-data
examples:
    generate training-data from mask-images

        >>> generator.Generator(datasetname, openForWriting=True)
        >>> gen.generate_dataset(path_to_images, full_labeled=True)

    show information about a dataset:

        >>> gen = generator.Generator(datasetname).show_details()

    load training-data, already splitted in training/testdata
    (ATTENTION: to train the current used CNN, you need to reformat data, see in trainerclass.preprocess_data)

        >>> gen = generator.Generator(datasetname)
        >>> X_train, X_test, y_train, y_test= gen.load_traindata()

"""

class Generator(object):
    def __init__(self, dataset_name, append=False, openForWriting = False):
        self.dataset_path = conf.TRAINING_DATASET_PATH+dataset_name+".hdf5"
        if (os.path.exists(self.dataset_path)):
            if(openForWriting):
                if (append):
                    self.f = h5py.File(self.dataset_path, 'r+')
                else:
                    os.remove(self.dataset_path)
                    self.f = h5py.File(self.dataset_path, "w")
            else:
                self.f = h5py.File(self.dataset_path, 'r')
        else:
            self.f = h5py.File(self.dataset_path, "w")

    def load_traindata(self, test_size=0.3, equal_class_sizes=True, dataset="", max_size_per_set=None):
        if len(dataset) > 0:
            if (not dataset in self.f):
                raise Exception("data-set not found in file")
            else:
                datasets = [dataset]
        else:
            datasets = []
            for name in self.f:
                datasets.append(name)

        X = []
        y = []

        for set in datasets:
            print("load samples from dataset " + set)
            dset = self.f.get(set)
            X_tmp, y_tmp = self._add_samples(dset, equal_class_sizes, max_size_per_set=max_size_per_set)
            X.extend(X_tmp)
            y.extend(y_tmp)
        print(" got total" + str(np.shape(X)) + " samples..")
        print(" generate training/test-data with test-size:" + str(test_size))
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
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
        if (max_size_per_set is not None):
            max_sample_size = int(np.round(max_size_per_set / len(conf.CLASS_LABEL_MAPPING)))
        elif (conf.NETWORK_TRAIN_MAX_SAMPLES is not None):
            max_sample_size = int(np.round(conf.NETWORK_TRAIN_MAX_SAMPLES / len(conf.CLASS_LABEL_MAPPING)))

        min_sample_count = -1
        if (equal_class_sizes):
            for name in dset:
                ds = dset.get(name)
                if name in conf.CLASS_LABEL_MAPPING.values():
                    if (min_sample_count == -1) | (ds.len() < min_sample_count):
                        min_sample_count = ds.len()
        if (max_sample_size is not None):
            if (max_sample_size < min_sample_count):
                min_sample_count = max_sample_size

        X = []
        y = []
        inv_class_labeling =  {v: k for k, v in conf.CLASS_LABEL_MAPPING.items()}
        # iterate over classes
        for name in dset:
            if name in conf.CLASS_LABEL_MAPPING.values():
                ds = dset.get(name)
                print('     add training-data for label: ' + name)

                values = ds[()]
                if (equal_class_sizes):
                    if (min_sample_count > 0):
                        values = values[np.random.choice(values.shape[0], min_sample_count), :]
                    else:
                        values = []
                    sample_count = min_sample_count
                else:
                    sample_count = ds.len()
                X.extend(values)
                print(str(np.shape(X)))

                encoded_class = inv_class_labeling.get(name)
                y.extend([encoded_class] * sample_count)
                print('     added ' + str(sample_count) + " samples")

        return X, y

    def show_details(self, dataset=""):
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
                datasets = [dataset]
        else:
            datasets = []
            for name in self.f:
                datasets.append(name)
        for set in datasets:
            self._print_details_for_dataset(set)

    def _print_details_for_dataset(self, dataset):
        print("meta-informations for " + dataset + ":")
        ds = self.f.get(dataset)
        for k in ds.attrs.keys():
            print("     "+str(k)+":"+str(ds.attrs[k]))
        print("sample-count details for " + dataset + ":")
        for name in self.f.get(dataset):
            dset = self.f.get(dataset).get(name)

            print("    " + name + ": " + str(dset.len()))

    def _generate_labels(self, mask):
        mask_mapped = np.copy(mask)
        labels = np.zeros((mask.shape[0], mask.shape[1], 8), dtype=np.uint8)
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                # = find_nearest(mask[i,j])
                mask_mapped[i, j], labels[i, j] = self._find_nearest(mask[i, j])
        return mask_mapped, labels

    def _label_generate_mask(self, labels):
        mask = np.zeros((labels.shape[0], labels.shape[1], 3), dtype=np.uint8)
        color_mapping_arr = np.asarray([v for k, v in conf.CLASS_COLOR_MAPPING.items()])
        for i in range(labels.shape[0]):
            for j in range(labels.shape[1]):
                mask[i, j] = np.dot(np.transpose(color_mapping_arr), labels[i, j])

        return mask

    def _find_nearest(self, value, encode_class=False):
        color_mapping_arr = [v for k, v in conf.CLASS_COLOR_MAPPING.items()]
        array = np.asarray(color_mapping_arr)
        value = np.asarray(value)
        distances = [np.linalg.norm(v - value) for v in array]
        idx = np.asarray(distances).argmin()
        v = np.zeros(8)
        if(conf.CLASS_MERGES is not None):
            if(conf.CLASS_MERGES.get(idx) is not None):
                idx = conf.CLASS_MERGES.get(idx)
        v[idx] = 1
        return color_mapping_arr[idx], v

    # color_mapping = {1:[255,255,0], 2: [100,100,100], 3:[100,100,0],  4: [255, 0, 0], 5:[0, 255, 0],  8: [0, 0, 255]}
    # class_mapping = {0:"bee_head", 1: "bee_breast", 2: "bee_back", 3: "tag", 4:"honey", 5:"comb"}

    def generate_masks_for_classes(self, fn):
        mask = imread(fn)
        if (conf.GENERATOR_COMPRESSION_FAKTOR is not None):
            targetsize = np.round(np.array(mask.shape) / conf.GENERATOR_COMPRESSION_FAKTOR).astype(int)
            mask = imresize(mask, (targetsize[0], targetsize[1]))

        masked_mapped, masks = self._generate_labels(mask)
        masks = np.transpose(masks, axes=(2, 0, 1))
        return masks

    def generate_dataset(self, image_path, full_labeled=False, mask_type="",
                         compression=conf.GENERATOR_COMPRESSION_FAKTOR):
        # check if path is directory or file,
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

        # check mask_types
        if len(mask_type) > 0:
            if mask_type in conf.CLASS_LABEL_MAPPING.items():
                mask_types = [mask_type]
            else:
                raise Exception("mask-type " + mask_type + " not found")
        else:
            mask_types = conf.CLASS_LABEL_MAPPING.items()

        # iterate over all found files
        for file in files:
            image_base, extention = os.path.splitext(os.path.basename(file))
            dataset_group = image_base
            if dataset_group in self.f:
                answer = self._user_yes_no_query(
                    "image-samples for " + dataset_group + " already exits. should image be skipped?")
                if answer == 1:
                    continue

            grp = self.f.require_group(dataset_group)
            grp.attrs['included_classes']= str(conf.CLASS_LABEL_MAPPING)
            grp.attrs['shift']= str(conf.GENERATOR_SHIFT)
            grp.attrs['min_overlapping_small']= str(conf.GENERATOR_MIN_OVERLAPPING_SMALL)
            grp.attrs['min_overlapping']= str(conf.GENERATOR_MIN_OVERLAPPING)
            grp.attrs['min_overlapping_big']= str(conf.GENERATOR_MIN_OVERLAPPING_BIG)
            #  if the image is full labeled, generate masks out of labeled-image
            if (full_labeled):
                grp.attrs['full_labeled'] = "YES"
                image_base, extention = os.path.splitext(os.path.basename(file))
                image_path, img = os.path.split(os.path.abspath(file))
                fn = image_path + "/" + image_base + '_' + 'full_labeled' + '.jpg'
                masks = self.generate_masks_for_classes(os.path.realpath(fn))
                for mask_type in mask_types:
                    im = masks[mask_type[0]]
                    self._generate_samples_for_mask(grp, os.path.realpath(file), mask_type[1], accept_outside=False,
                                                    compression=compression, im_mask=im)
            else:
                grp.attrs['full_labeled'] = "NO"
                for mask_type in mask_types:
                    self._generate_samples_for_mask(grp, os.path.realpath(file), mask_type[1], accept_outside=False,
                                                    compression=compression)

    def _generate_samples_for_mask(self, grp, file, mask_type, im_mask=None, accept_outside=False, compression=None):
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
        name = grp.name
        if (mask_type in grp):
            print("nothing to do in " + name + "/" + mask_type + ", type exists")
            return
        print("start processing " + name)
        output = conf.GENERATOR_OUTPUT | conf.ANALYSE_PLOTS_SAVE

        samples = []

        orig_size = conf.GENERATOR_ORIG_SAMPLE_SIZE
        max_shift = conf.GENERATOR_SHIFT

        # if compression is activated, target size will be compressed
        if (compression is not None):
            sample_size = [int(np.round(orig_size[0] / compression)), int(np.round(orig_size[1] / compression))]
        else:
            sample_size = orig_size

        # size for target-vector
        vector_size = sample_size[0] * sample_size[1]

        print("generate samples for mask-type:" + mask_type + " with orig-smaple-size:" + str(
            orig_size) + " max-shift:" + str(max_shift) + "and compression:" + str(compression) + ".")
        print("new size of samples:" + str(sample_size) + " and as vector:" + str(vector_size))
        grp.attrs['compression_factor'] = compression
        grp.attrs['orig_sample_size'] = orig_size
        grp.attrs['compr_sample_size'] = sample_size



        # if mask is not given as parameter, try to find it in file-system
        if im_mask is None:
            image_base, extention = os.path.splitext(os.path.basename(file))
            image_path, img = os.path.split(os.path.abspath(file))
            # target_directory = image_path+"/"+image_base+"_"+mask_type

            # print(name)
            # print(image_path+"/"+image_base+'_'+mask_type+'.png')
            # exit()
            im_mask = imread(image_path + "/" + image_base + '_' + mask_type + '.png', flatten=True)

            if (compression is not None):
                mask_compressed_size = (
                int(np.round(im_mask.shape[0] / compression)), int(np.round(im_mask.shape[1] / compression)))
                im_mask = imresize(im_mask, mask_compressed_size)
            # if (compression is not None):
            # @TODO compress whole image
            im_mask = stats.threshold(im_mask, threshmin=0.1, threshmax=10000, newval=0)
            im_mask = stats.threshold(im_mask, threshmin=0, threshmax=0.1, newval=1)

        orig = imread(file, flatten=True)

        grp.attrs['orig_image_size'] = str(orig.shape)

        if (compression is not None):
            orig_compressed_size = (
            int(np.round(orig.shape[0] / compression)), int(np.round(orig.shape[1] / compression)))
            orig = imresize(orig, orig_compressed_size)

        if (orig.shape != im_mask.shape):
            print("ERROR: mask(" + str(im_mask.shape) + ") and original-image(" + str(
                orig.shape) + ") has different sizes, shutdown")
            exit()

        grp.attrs['compressed_image_size'] = orig.shape

            # if not target == False:
            # target_directory = target_directory+"/"+mask_type

            # if not os.path.exists(target_directory):
            #   print("generate target folder "+target_directory)
            #   os.makedirs(target_directory)

        # print(image.shape)
        # print(image)
        # subimage =image[0:80,0:80]
        # x, y = np.ogrid[-np.pi:np.pi:100j, -np.pi:np.pi:100j]
        # r = np.sin(np.exp((np.sin(x)**3 + np.cos(y)**2)))

        if output:
            fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True,)
            fig2, ax2 = plt.subplots(nrows=6, ncols=6, sharex=True, sharey=True, )
            ax2 = ax2.flatten()
            #ax[0].imshow(im_mask, interpolation='nearest', cmap=plt.cm.gray)
            ax.imshow(orig, interpolation='nearest', cmap=plt.cm.gray)

        possible_indices = np.transpose(np.nonzero(im_mask))
        all_sample_count = 0
        i = 1
        max_sample_count = possible_indices.shape[0]
        max_machting = (sample_size[0] * sample_size[1])
        big_mask = False
        contour_size = None
        thres = conf.GENERATOR_MIN_OVERLAPPING
        if not "bee" in mask_type:
            big_mask = True
            thres = conf.GENERATOR_MIN_OVERLAPPING_BIG
            print("big mask detected, start shifting..")
        elif "tag" in mask_type or "middle" in mask_type or "bee_in_comb":
            contour_size = int(np.round(50/compression)**2)
            print("small contours detected, using " +str(contour_size)+" for matching")
        elif "head" in mask_type:
            contour_size = int(np.round(60/compression))
            print("small contours detected, using " +str(contour_size)+" for matching")

        vals = []
        # iterate over all positions
        for current_position in possible_indices:
            # if position is in shift, ignore
            if ((i % conf.GENERATOR_SHIFT) == 0)| (not big_mask):
                #print(current_position)
                start_x = int(np.round(current_position[0] - sample_size[0] / 2))
                start_y = int(round(current_position[1] - sample_size[1] / 2))

                if (start_x >= 0) & (start_y >= 0) & (start_x + sample_size[0] <= orig.shape[0]) & \
                        (start_y + sample_size[1] <= orig.shape[1]):
                    # print(start_x)
                    # print(start_y)
                    # print(sample_size[0])
                    # print(sample_size[1])

                    sliding_window = im_mask[start_x:(start_x + sample_size[0]), start_y:(start_y + sample_size[1])]
                    vals.append(np.sum(sliding_window))
                    # print(sliding_window)
                    # print(np.sum(sliding_window))
                    # exit()
                    # print(np.sum(sliding_window)/max_machting)
                    # check if windows is overlapping enough
                    #print(((np.sum(sliding_window) / max_machting) >= thres) | ((contour_size is not None)  & (np.sum(sliding_window) >= contour_size)))
                    #print(np.sum(sliding_window) >= contour_size)
                    #print(contour_size is not None)
                    #print((contour_size is not None)  & (np.sum(sliding_window) >= contour_size))
                    add = False
                    if ((np.sum(sliding_window) / max_machting) >= thres):
                        add = True
                    elif contour_size is not None :
                        if np.sum(sliding_window) >= contour_size:
                            add = True
                    if add:
                        sample = orig[start_x:(start_x + sample_size[0]), start_y:(start_y + sample_size[1])]

                        # if output  is activated draw frame and add sample to output
                        if output:
                            x_1 = [start_x, start_y]
                            x_2 = [start_x, start_y + sample_size[1]]
                            y_2 = [start_x + sample_size[0], start_y]
                            y_1 = [start_x + sample_size[0], start_y + sample_size[1]]
                            #ax.plot([x_1[1], x_2[1], y_1[1], y_2[1], x_1[1]], [x_1[0], x_2[0], y_1[0], y_2[0], x_1[0]],
                                   # linewidth=1)
                            ax.plot([x_1[1], x_2[1], y_1[1], y_2[1], x_1[1]], [x_1[0], x_2[0], y_1[0], y_2[0], x_1[0]],
                                    linewidth=1)
                                #ax2[all_sample_count].set_title('%d_%d' % (start_x, start_y))

                        sample_vector = sample.reshape(vector_size)
                        samples.append(sample_vector)
                        all_sample_count += 1
                    else:
                        if output:
                            x_1 = [start_x, start_y]
                            x_2 = [start_x, start_y + sample_size[1]]
                            y_2 = [start_x + sample_size[0], start_y]
                            y_1 = [start_x + sample_size[0], start_y + sample_size[1]]
                            #ax.plot([x_1[1], x_2[1], y_1[1], y_2[1], x_1[1]], [x_1[0], x_2[0], y_1[0], y_2[0], x_1[0]],
                                   # linewidth=1)
                            #ax.plot([x_1[1], x_2[1], y_1[1], y_2[1], x_1[1]], [x_1[0], x_2[0], y_1[0], y_2[0], x_1[0]],
                                 #   '-.', linewidth=1)
            if(len(vals)== 200):
                print(vals)
            i += 1
            sys.stderr.write('\r %d/%d:, total for mask-type: %d  ' % (i, max_sample_count, all_sample_count))
            sys.stderr.flush()

        print("finished, found " + str(all_sample_count) + " samples to use")

        if output:
            if all_sample_count >= 36:
                indexes = np.random.choice(len(samples), 36)
                i = 0
                for ind in indexes:
                    #print(ind)
                    ex = samples[ind]
                    ax2[i].imshow(ex.reshape((sample_size[0], sample_size[1])), cmap=plt.cm.gray, interpolation='nearest')
                    i += 1
            ax.set_xticks([])
            ax.set_yticks([])
            #ax1.set_xticks([])
            #ax1.set_yticks([])
            ax2[0].set_xticks([])
            ax2[0].set_yticks([])
            plt.tight_layout()
            if(conf.ANALYSE_PLOTS_SAVE):
                fn,ext1 = os.path.splitext(os.path.basename(file))
                fnm = conf.ANALYSE_PLOTS_PATH+fn+'_'+mask_type+'_mask.png'
                fig.savefig(fnm)
                fns = conf.ANALYSE_PLOTS_PATH+fn+'_'+mask_type+'_samples.png'
                fig2.savefig(fns)
            if(conf.GENERATOR_OUTPUT):
                plt.show()

        dset = grp.create_dataset(mask_type, data=samples)
        dset.attrs['class_label'] = mask_type



        # contours = measure.find_contours(r, 0)
        # print("found "+ str(len(contours))+" polygons")
        # Display the image and plot all contours found
        #
        # all_sample_count = 0
        # # plt.savefig('./figures/mnist_miscl.png', dpi=300)
        #
        # for n,contour in enumerate(contours):
        #     print('process contour '+str(n)+'...')
        #     if output:
        #         ax1.plot(contour[:, 1], contour[:, 0], linewidth=2)
        #         #ax[1].plot(contour[:, 1], contour[:, 0], linewidth=2)
        #
        #     bbBox = contour#np.transpose([contour[:,0], contour[:,1]])
        #     #if output:
        #         #ax[0].plot(bbBox[:, 1], bbBox[:, 0], linewidth=2)
        #     bbPath = mplPath.Path(bbBox, closed=True)
        #
        #     #get outer bounding box
        #     max_x = np.max(bbBox[:,0], axis=0)
        #     max_y = np.amax(bbBox[:,1], axis=0)
        #
        #     min_x = np.amin(bbBox[:,0], axis=0)
        #     min_y = np.amin(bbBox[:,1], axis=0)
        #
        #     #if output:
        #         #ax[0].plot([min_y, max_y, max_y, min_y, min_y], [min_x,min_x,max_x,max_x, min_x],linewidth=2)
        #
        #     #if(min_x +size[0] > max_x):
        #     #    samples_count = 1
        #     #    single_sample = True
        #     #else:
        #     #    samples_count = np.round(((max_x-min_x))*((max_y-min_y))*ratio, 0)
        #     #    single_sample = False
        #
        #
        #     single_sample_col = False
        #     sample_start_x = min_x
        #     samples_in_cols = 0
        #     if accept_outside & (sample_start_x > max_x-sample_size[0]):
        #         single_sample_col = True
        #         offset_x = np.round((sample_start_x - max_x-sample_size[0])/2)
        #         sample_start_x -= offset_x
        #     while sample_start_x <= (max_x-sample_size[0]):
        #         single_sample_row = False
        #         positions_y =[pos[1] for pos in contour if pos[0] == sample_start_x]
        #         y_min_tmp = np.amin(positions_y, axis=0)
        #         y_max_tmp = np.amax(positions_y, axis=0)
        #         sample_start_y = y_min_tmp
        #         if (y_max_tmp- sample_size[1] < y_min_tmp) & accept_outside:
        #             single_sample_row = True
        #             offset_y = np.round((sample_start_y - y_max_tmp-sample_size[1])/2)
        #             sample_start_y -= offset_y
        #         samples_in_row = 0
        #         while (sample_start_y <= (y_max_tmp-sample_size[0])) | single_sample_row:
        #             #bbPath = mplPath.Path([[min_x,min_y],[min_x,max_y],[max_x,max_y],[max_x,min_y],[min_x,min_y]], closed=True)
        #             x_1 = [sample_start_x, sample_start_y]
        #             x_2 = [sample_start_x, sample_start_y+sample_size[1]]
        #             y_2 = [sample_start_x+sample_size[0], sample_start_y]
        #             y_1 = [sample_start_x+sample_size[0], sample_start_y+sample_size[1]]
        #             rect =[x_1,x_2,y_1,y_2]
        #
        #             #sampleBox = trans.Bbox.from_bounds(sample_start_x[i], sample_start_y[i],80,80)
        #             #if i< 10:
        #                 #print(rect)
        #             res =bbPath.contains_points(rect)
        #
        #             bbPath.unit_rectangle()
        #             #if i <10:
        #                 #print(res)
        #
        #             #if bbPath.intersects_bbox(sampleBox, filled=True):
        #
        #             #if sample is fully in bound box add
        #             if res.all() | bees_mask:
        #                 samples_in_row += 1
        #                 subimage = orig[sample_start_x:(sample_start_x+sample_size[0]),sample_start_y:(sample_start_y+sample_size[1])]
        #                 #if(compression is not None):
        #                     #subimage = cv2.resize(subimage,(compressed_compressed_size[0], compressed_compressed_size[1]))
        #
        #                     #plt.imsave(target_directory+"/"+image_base+"_sample_"+str(sample_start_x)+"_"+str(sample_start_y)+"_"+mask_type+".jpg",subimage,cmap=plt.cm.gray)
        #                 all_sample_count += 1
        #                 if output:
        #                     ax.plot([x_1[1], x_2[1], y_1[1], y_2[1], x_1[1]],[x_1[0],x_2[0],y_1[0],y_2[0], x_1[0]],  linewidth=2)
        #             #else:
        #                 #print('no match')
        #                 #ax.plot( [x_1[1], x_2[1], y_1[1], y_2[1], x_1[1]],[x_1[0],x_2[0],y_1[0],y_2[0], x_1[0]],'-.',    linewidth=2)
        #             if single_sample_row:
        #                 break
        #             else:
        #                 sample_start_y += np.random.randint(1, max_shift)
        #         samples_in_cols += samples_in_row
        #
        #         sample_start_x += np.random.randint(1, max_shift)
        #         samples_in_row = 0
        #         if(single_sample_col):
        #             break
        #     print('done')
        #
        # if output:
        #     #ax2[0].set_xticks([])
        #     #ax2[0].sets_yticks([])
        #     #plt.tight_layout()
        #     plt.show()
        #     print("generated "+ str(all_sample_count)+" samples for type "+mask_type)
        #     print(str(np.shape(samples)))
        # dset = grp.create_dataset(mask_type,data=samples)
        # #dset.attrs['class_label'] = mask_type
        # print("")

    def _user_yes_no_query(self, question):
        sys.stdout.write('%s [y/n]\n' % question)
        while True:
            try:
                return strtobool(input().lower())
            except ValueError:
                sys.stdout.write('Please respond with \'y\' or \'n\'.\n')
