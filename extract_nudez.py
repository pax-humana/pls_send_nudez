#!/usr/bin/env python3

import argparse
import string
import os
import sys
import numpy
from PIL import Image
from io import BytesIO

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Suppress Tensorflow log messages
import opennsfw2

NSFW_MODEL_DIR = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--threshold', type=float, default=.7, help='Default matching threshold for the open_nsfw model (0 - 1). Default: .7"')
parser.add_argument('directory', nargs='?', type=str, default='./', help='Input Directory. Default: current working directory.')
parser.add_argument('output', nargs='?', type=str, default='wins', help='Output Subdirectory. Default: wins')
args = parser.parse_args()

SCORE_THRESHOLD = args.threshold
INPUT_DIR = os.path.abspath(args.directory)
OUTPUT_DIR = os.path.abspath(os.path.join(INPUT_DIR, args.output))
EXCLUDE = set([args.output])

def preprocess_and_compute(pimg, model):
    image = opennsfw2.preprocess_image(Image.open(BytesIO(pimg)), opennsfw2.Preprocessing.YAHOO)
    inputs = numpy.expand_dims(image, axis=0)
    return model.predict(inputs)

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Pre-load Tensorflow OpenNSFW2 model
    model = opennsfw2.make_open_nsfw_model()

    for fulldir, subdir, files in os.walk(INPUT_DIR):
        subdir[:] = [d for d in subdir if d not in EXCLUDE] # Exclude the output directory
        for filename in files:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp', '.gif')):
                try:
                    image_data = open(os.path.join(fulldir, filename), 'rb').read()
                    scores = preprocess_and_compute(image_data, model)
                except:
                    continue
                if scores[0][1] >= SCORE_THRESHOLD:
                    print(os.path.join(fulldir, filename), "NSFW score:" , scores[0][1])
                    os.rename(os.path.join(fulldir, filename), os.path.join(OUTPUT_DIR, filename))
            
            else:
                print(os.path.join(INPUT_DIR, filename), "is not a supported image.")

if __name__ == '__main__':
    main()
