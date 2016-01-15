import combdetection.config as conf
if not conf.GENERATOR_OUTPUT:
    #needed to plot images on flip
    import matplotlib
    matplotlib.use('Agg')

import matplotlib.pyplot as plt
import sys
import combdetection.utils.generator as generator

import combdetection.neuralnet
import numpy as np



if __name__ == '__main__':

    directory = sys.argv[2]
    target = sys.argv[1]

    gen = generator.Generator(target, openForWriting=True)
    gen.generate_dataset(directory, full_labeled=conf.GENERATOR_FULL_LABELED)
