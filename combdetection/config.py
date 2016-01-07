#!/usr/bin/python3

GENERATOR_SAMPLE_SIZE= (110,110)
#GENERATOR_SAMPLE_SIZE= (28,28)
GENERATOR_COMPRESSION_FAKTOR = 6
GENERATOR_MAX_SHIFT = 10
GENERATOR_OUTPUT= False
NETWORK_CLASS_LABELS= {'empty':0,'filled':1,'beesback':2}
NETWORK_SAMPLE_SIZE= (18,18)
NETWORK_TRAIN_MAX_SAMPLES = None
#filtersize = (16, 16)
filenames_mmapped = {name: '{}.mmapped'.format(name) for name in
                     ['xtrain', 'ytrain', 'xval', 'yval', 'xtest', 'ytest']}

