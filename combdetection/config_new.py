




"""
SMALL NETWORK FOR COMB-CLASSIFYING
"""
NETWORK_WEIGHTS_PATH= "./weights/"

NETWORK_COMBS_WEIGHTS_FILE_NAME="nn_ds_bees_noclahe_c4_2"

# Colors for displaying classes
NETWORK_COMBS_CLASS_COLOR_MAPPING = { 0: [255, 0, 0],1: [0, 255, 0], 2: [0, 0, 255]}

# Labels for classes
NETWORK_COMBS_CLASS_LABEL_MAPPING = {0: "bees",1: "filled_comb", 2: "empty_comb"}

# minimal accepted size of a bee contour in original image
BEEDETECTOR_MIN_BEE_AREA = 50*50

# maximal accepted size of a bee contour in original image
BEEDETECTOR_MAX_BEE_AREA = 230*85

# should be chosen carefully, because bee might be bent
BEEDETECTOR_MAX_WIDTH =250

# always the bigger axis  is taken here
BEEDETECTOR_MAX_HEIGHT = 130

HIVE_WATCHER_COMB_SAMPLES = 5

class_label_mapping_german = {0: "Tag (Biene)",1: "gedeckelte Wabe", 2: "ungedeckelte Wabe",3: "Mittelteil (Biene)", 4: "Kopf (Biene)", 5: "Hinterteil (Biene)",
                 6: "Biene komplett in Wabe"}