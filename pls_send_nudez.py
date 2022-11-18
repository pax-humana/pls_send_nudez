#!/usr/bin/env python3

import argparse
import string
import random
import requests
import hashlib
import os
import sys
import numpy
from PIL import Image
from io import BytesIO

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Suppress Tensorflow log messages
import opennsfw2

# Defaults
IMAGE_SIZE_MIN = 1024 * 20 # Minimum filesize (20k) for downloaded images.
IMGUR_URL_HASH_LENGTH = 5 # Default imgur URL hash length. Should be 5 or 7.
NSFW_MODEL_DIR = os.path.dirname(os.path.abspath(__file__))

# Constants
IMAGE_EXTENSION = ".jpg" # Extension for search.
IMGUR_URL_PREFIX = "http://i.imgur.com/" # Prefix for Imgur URLs.

# Process command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('-s', '--min-size', type=int, default=IMAGE_SIZE_MIN, help='Minimum image size in kilobytes. (Default: ' + str(IMAGE_SIZE_MIN) + ')')
parser.add_argument('-7', '--new-hash-length', action='store_true', help='Use image filename length of 7. (Default is 5)')
parser.add_argument("output", type=str, help='Output folder name.')
args = parser.parse_args()

if args.min_size is not IMAGE_SIZE_MIN:
    IMAGE_SIZE_MIN = args.min_size * 1024

if args.new_hash_length is True:
    IMGUR_URL_HASH_LENGTH = 7

OUTPUT_DIR = os.path.abspath(args.output)

# Functions
def gen_random_img_hash(length):
    return ''.join(random.SystemRandom().choice(string.ascii_letters + string.digits) for _ in range(length))
    #return "4iboqds" # A nice pussy for testing
    #return "BgzZmlo" # An image to train on

def get_image(image_name):
    image_url = IMGUR_URL_PREFIX + image_name
    req = requests.get(image_url)
    image_data = req.content
    if 'd835884373f4d6c8f24742ceabe74946' == hashlib.md5(image_data).hexdigest() or sys.getsizeof(image_data) < IMAGE_SIZE_MIN:
        return None
    else:
        return image_data

def preprocess_and_compute(pimg, model):
    image = opennsfw2.preprocess_image(Image.open(BytesIO(pimg)), opennsfw2.Preprocessing.YAHOO)
    inputs = numpy.expand_dims(image, axis=0)
    return model.predict(inputs)

# Main
def main():
    # Pre-load Tensorflow OpenNSFW2 model
    print("Loading OpenNSFW2 model")
    model = opennsfw2.make_open_nsfw_model()

    # Create output directory
    if not os.path.isdir(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Find images and apply the model
    print("Starting the scraper")
    while True:
        # Get a random image
        image_data = None
        while image_data is None:
            image_name = gen_random_img_hash(IMGUR_URL_HASH_LENGTH) + IMAGE_EXTENSION
            image_data = get_image(image_name)
        image_path = os.path.join(os.sep, OUTPUT_DIR, image_name)

        # Classify downloaded image
        try:
            scores = preprocess_and_compute(image_data, model) 
        except:
            print("ERR! " + image_name)
            continue

        # Save wins
        if scores[0][1] >= .7:
            print("WIN! " + image_name)
            with open(image_path, "wb") as f:
                f.write(image_data)

if __name__ == '__main__':
    main()
