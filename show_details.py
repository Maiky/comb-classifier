import sys
import combdetection.utils.generator as generator
import combdetection.neuralnet
import numpy as np



if __name__ == '__main__':

    target = sys.argv[1]

    gen = generator.Generator(target)
    gen.show_details()
