#!/usr/bin/python3

#
#   GENERATOR PARAMETER
#

# size of training-data sample size before compression
GENERATOR_ORIG_SAMPLE_SIZE= (90, 90)

# compression factor for downscaling
GENERATOR_COMPRESSION_FAKTOR = 2

# mininum percentage, for that a generated sample must be overlapped to the original contour
GENERATOR_MIN_OVERLAPPING= 0.7
GENERATOR_MIN_OVERLAPPING_BIG= 0.8
# defines the shift-offset of the sliding window of the generator-image
GENERATOR_SHIFT = 10

# if set to true, plots for sample-generation will be shown
GENERATOR_OUTPUT= False
GENERATOR_FULL_LABELED = False

#
#   TRAINING PARAMETER
#
TRAINING_LOG_SAVE = True
TRAINING_DATASET_PATH = './datasets/'
TRAINING_LOG_PATH = './logs/'
TRAINING_WEIGHTS_PATH = './saved_networks/'

ANALYSE_PLOTS_PATH= './plots/'
ANALYSE_PLOTS_SAVE= True
ANALYSE_PLOTS_SHOW = False


NETWORK_SAMPLE_SIZE= (22,22)

NETWORK_TRAIN_MAX_SAMPLES = None
#CLASS_COLOR_MAPPING = { 2: [255, 255, 0]}
#CLASS_LABEL_MAPPING = {2: "bee_head"}
CLASS_COLOR_MAPPING = { 0: [255, 0, 0],1: [0, 255, 0], 2: [0, 0, 255], 3: [240, 157, 157], 4: [255, 255, 0], 5: [255, 100, 0],
                 6: [92, 208, 205], 7: [100, 100, 100]}
CLASS_MERGES = {3:4,5:4,2:4}
#CLASS_LABEL_MAPPING = {0: "bee_tag",1: "filled_comb", 2: "empty_comb"}#, 3: "bee_middle", 4: "bee_head", 5: "bee_back",
                 #6: "bee_in_comb", 7: "undef"}
CLASS_LABEL_MAPPING = {0: "beesback",1: "filled", 2: "empty"}

