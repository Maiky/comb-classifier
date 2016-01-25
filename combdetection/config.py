#!/usr/bin/python3

#
#   GENERATOR PARAMETER
#

# size of training-data sample size before compression
GENERATOR_ORIG_SAMPLE_SIZE= (90, 90)

# compression factor for downscaling
GENERATOR_COMPRESSION_FAKTOR = 6

# mininum percentage, for that a generated sample must be overlapped to the original contour
GENERATOR_MIN_OVERLAPPING_SMALL= 0.4
GENERATOR_MIN_OVERLAPPING= 0.7
GENERATOR_MIN_OVERLAPPING_BIG= 0.4
# defines the shift-offset of the sliding window of the generator-image
GENERATOR_SHIFT = 100

GENERATOR_BEE_MERGE=True

# if set to true, plots for sample-generation will be shown
GENERATOR_OUTPUT= False
GENERATOR_FULL_LABELED = True

GENERATOR_USE_CLAHE = False
#
#   TRAINING PARAMETER
#
TRAINING_LOG_SAVE = True
TRAINING_DATASET_PATH = './datasets/'
TRAINING_LOG_PATH = './logs/'
TRAINING_WEIGHTS_PATH = './saved_networks/'

ANALYSE_PLOTS_PATH= './plots/'
ANALYSE_PLOTS_SAVE= True
ANALYSE_PLOTS_SHOW = True


NETWORK_SAMPLE_SIZE= (15,15)#(22,22)

NETWORK_TRAIN_MAX_SAMPLES = None
#CLASS_COLOR_MAPPING = { 2: [255, 255, 0]}
#CLASS_LABEL_MAPPING = {2: "bee_head"}
#CLASS_COLOR_MAPPING = { 0: [255, 0, 0],1: [0, 255, 0], 2: [0, 0, 255], 3: [240, 157, 157], 4: [255, 255, 0], 5: [255, 100, 0],
               #  6: [92, 208, 205]}# 7: [100, 100, 100]}
CLASS_MERGES = None#{3:0,5:0,4:0}


class_color_mapping_cl3 = { 0: [255, 0, 0],1: [0, 255, 0], 2: [0, 0, 255]}

class_color_mapping_cl7 = { 0: [255, 0, 0],1: [0, 255, 0], 2: [0, 0, 255], 3: [240, 157, 157], 4: [255, 255, 0], 5: [255, 100, 0],
                 6: [92, 208, 205]}

class_label_mapping_cl3= {0: "bee",1: "filled_comb", 2: "empty_comb"}

class_label_mapping_cl7= {0: "bee_tag",1: "filled_comb", 2: "empty_comb",3: "bee_middle", 4: "bee_head", 5: "bee_back",
                 6: "bee_in_comb"}



# minimal accepted size of a bee contour in original image
BEEDETECTOR_MIN_BEE_AREA = 50*50

# maximal accepted size of a bee contour in original image
BEEDETECTOR_MAX_BEE_AREA = 230*85

# should be chosen carefully, because bee might be bent
BEEDETECTOR_MAX_WIDTH =250

# always the bigger axis  is taken here
BEEDETECTOR_MAX_HEIGHT = 130

HIVE_WATCHER_COMB_SAMPLES = 5

if GENERATOR_BEE_MERGE:
    CLASS_COLOR_MAPPING = class_color_mapping_cl3
    CLASS_LABEL_MAPPING = class_label_mapping_cl3
else:
    CLASS_COLOR_MAPPING = class_color_mapping_cl7
    CLASS_LABEL_MAPPING = class_label_mapping_cl7

class_label_mapping_german = {0: "Tag (Biene)",1: "gedeckelte Wabe", 2: "ungedeckelte Wabe",3: "Mittelteil (Biene)", 4: "Kopf (Biene)", 5: "Hinterteil (Biene)",
                 6: "Biene komplett in Wabe"}
#CLASS_LABEL_MAPPING = {0: "beesback",1: "filled", 2: "empty"}

