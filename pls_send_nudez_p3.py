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

os.environ['GLOG_minloglevel'] = '3' # supress Caffe verbose prints
import caffe

# Defaults
IMAGE_SIZE_MIN = 1024 * 20 # Minimum filesize (20k) for downloaded images.
IMGUR_URL_HASH_LENGTH = 5 # Default imgur URL hash length. Should be 5 or 7.
NSFW_MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
# GPU_ID = 0

# Constants
IMAGE_EXTENSION = ".jpg" # Extension for search.
IMGUR_URL_PREFIX = "http://i.imgur.com/" # Prefix for Imgur URLs.

# Process command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('-s', '--min-size', type=int, default=IMAGE_SIZE_MIN, help='Minimum image size in kilobytes. (Default: ' + str(IMAGE_SIZE_MIN) + ')')
parser.add_argument('-7', '--new-hash-length', action='store_true', help='Use image filename length of 7. (Default is 5)')
parser.add_argument('-n', '--nsfw-model', type=str, default=NSFW_MODEL_DIR, help='Location of the open_nsfw model root directory. Default: ' + str(NSFW_MODEL_DIR) + ')')
parser.add_argument("output", type=str, help='Output folder name.')
args = parser.parse_args()

if args.min_size is not IMAGE_SIZE_MIN:
    IMAGE_SIZE_MIN = args.min_size * 1024

if args.new_hash_length is True:
    IMGUR_URL_HASH_LENGTH = 7

NSFW_MODEL_DIR = os.path.abspath(args.nsfw_model)
OUTPUT_DIR = os.path.abspath(args.output)

# Functions
def gen_random_img_hash(length):
    return ''.join(random.SystemRandom().choice(string.ascii_letters + string.digits) for _ in range(length))
    #return "4iboqds" # A nice pussy for testing

def get_image(image_name):
    image_url = IMGUR_URL_PREFIX + image_name
    req = requests.get(image_url)
    image_data = req.content
    if 'd835884373f4d6c8f24742ceabe74946' == hashlib.md5(image_data).hexdigest() or sys.getsizeof(image_data) < IMAGE_SIZE_MIN:
        return None
    else:
        return image_data

def resize_image(data, sz=(256, 256)):
    im = Image.open(BytesIO(data))
    if im.mode != "RGB":
        im = im.convert('RGB')
    imr = im.resize(sz, resample=Image.BILINEAR)
    fh_im = BytesIO()
    imr.save(fh_im, format='JPEG')
    fh_im.seek(0)
    return fh_im

def caffe_preprocess_and_compute(pimg, caffe_transformer=None, caffe_net=None, output_layers=None):
    if caffe_net is not None:

        # Grab the default output names if none were requested specifically.
        if output_layers is None:
            output_layers = caffe_net.outputs

        img_bytes = resize_image(pimg, sz=(256, 256))
        image = caffe.io.load_image(img_bytes)

        H, W, _ = image.shape
        _, _, h, w = caffe_net.blobs['data'].data.shape
        h_off = max((H - h) / 2, 0)
        w_off = max((W - w) / 2, 0)
        crop = image[int(h_off):int(h_off + h), int(w_off):int(w_off + w), :]
        transformed_image = caffe_transformer.preprocess('data', crop)
        transformed_image.shape = (1,) + transformed_image.shape

        input_name = caffe_net.inputs[0]
        all_outputs = caffe_net.forward_all(blobs=output_layers,
            **{input_name: transformed_image})

        outputs = all_outputs[output_layers[0]][0].astype(float)
        return outputs
    else:
        return []

# Main
def main():
    # Pre-load open_nsfw caffe model from script directory
    nsfw_net = caffe.Net(os.path.join(os.sep, NSFW_MODEL_DIR, "nsfw_model", "deploy.prototxt"),
        os.path.join(os.sep, NSFW_MODEL_DIR, "nsfw_model", "resnet_50_1by2_nsfw.caffemodel"),
        caffe.TEST)

    # Load transformer
    # Note that the parameters are hard-coded for best results
    # caffe.set_device(GPU_ID)
    # caffe.set_mode_gpu()
    caffe_transformer = caffe.io.Transformer({'data': nsfw_net.blobs['data'].data.shape})
    caffe_transformer.set_transpose('data', (2, 0, 1))  # move image channels to outermost
    caffe_transformer.set_mean('data', numpy.array([104, 117, 123]))  # subtract the dataset-mean value in each channel
    caffe_transformer.set_raw_scale('data', 255)  # rescale from [0, 1] to [0, 255]
    caffe_transformer.set_channel_swap('data', (2, 1, 0))  # swap channels from RGB to BGR

    while True:
        # Get a random image
        image_data = None
        while image_data is None:
            # image_name = gen_random_img_hash(random.choice([5, 7])) + ".jpg"
            image_name = gen_random_img_hash(IMGUR_URL_HASH_LENGTH) + IMAGE_EXTENSION
            image_data = get_image(image_name)
        image_path = os.path.join(os.sep, OUTPUT_DIR, image_name)
        # print("TRY: " + image_name)

        # Classify downloaded image
        try:
            scores = caffe_preprocess_and_compute(image_data, caffe_transformer=caffe_transformer, caffe_net=nsfw_net, output_layers=['prob'])
        except:
            print("ERR! " + image_name)
            continue

        # Save wins
        if scores[1] >= .7:
            print("WIN! " + image_name)
            with open(image_path, "wb") as f:
                f.write(image_data)

if __name__ == '__main__':
    main()
