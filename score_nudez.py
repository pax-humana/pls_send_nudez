#!/usr/bin/env python3

import argparse
import string
import random
import os
import sys
import numpy
from PIL import Image
from io import BytesIO

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Suppress Tensorflow log messages
import opennsfw2

NSFW_MODEL_DIR = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument('directory', nargs='?', type=str, default='./', help='Input Directory. Default: current working directory.')
args = parser.parse_args()

INPUT_DIR = os.path.abspath(args.directory)

def preprocess_and_compute(pimg, model):
    image = opennsfw2.preprocess_image(Image.open(BytesIO(pimg)), opennsfw2.Preprocessing.YAHOO)
    inputs = numpy.expand_dims(image, axis=0)
    return model.predict(inputs)

def main():
    # Pre-load Tensorflow OpenNSFW2 model
    print("Loading OpenNSFW2 model")
    model = opennsfw2.make_open_nsfw_model()

    for filename in os.listdir(INPUT_DIR):
        if not filename.startswith("."):
            if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png") or filename.endswith(".gif"):
                image_data = open(os.path.join(INPUT_DIR, filename), 'rb').read()
#                with open(os.path.join(INPUT_DIR, filename), 'rb') as f:
#                    image_data = f.read()
                scores = preprocess_and_compute(image_data, model)
                print(filename, "NSFW score:" , scores[0][1])
            else:
                print(os.path.join(INPUT_DIR, filename), "is not a supported image.")

if __name__ == '__main__':
    main()
