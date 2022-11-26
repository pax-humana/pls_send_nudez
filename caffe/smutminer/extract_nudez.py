#!/usr/bin/python3

import argparse
import string
import os
import sys
import numpy
from PIL import Image
from io import BytesIO

os.environ['GLOG_minloglevel'] = '3' # supress Caffe verbose prints
import caffe # Caffe must be in your shell's PYTHONPATH, eg. "export PYTHONPATH=~/src/caffe/python:$PYTHONPATH"

NSFW_MODEL_DIR = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--nsfw-model', type=str, default=NSFW_MODEL_DIR, help='Location of the open_nsfw model root directory. Default: ' + str(NSFW_MODEL_DIR) + ')')
parser.add_argument('-t', '--threshold', type=float, default=.7, help='Default matching threshold for the open_nsfw model (0 - 1). Default: .7"')
parser.add_argument('directory', nargs='?', type=str, default='./', help='Input Directory. Default: current working directory.')
parser.add_argument('output', nargs='?', type=str, default='wins', help='Output Subdirectory. Default: wins')
args = parser.parse_args()

NSFW_MODEL_DIR = os.path.abspath(args.nsfw_model)
SCORE_THRESHOLD = args.threshold
INPUT_DIR = os.path.abspath(args.directory)
OUTPUT_DIR = os.path.abspath(os.path.join(INPUT_DIR, args.output))
EXCLUDE = set([args.output])

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

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Pre-load open_nsfw caffe model from script directory
    nsfw_net = caffe.Net(os.path.join(os.sep, NSFW_MODEL_DIR, "nsfw_model", "deploy.prototxt"),
        os.path.join(os.sep, NSFW_MODEL_DIR, "nsfw_model", "resnet_50_1by2_nsfw.caffemodel"),
        caffe.TEST)

    # Load transformer
    # Note that the parameters are hard-coded for best results
    caffe_transformer = caffe.io.Transformer({'data': nsfw_net.blobs['data'].data.shape})
    caffe_transformer.set_transpose('data', (2, 0, 1))  # move image channels to outermost
    caffe_transformer.set_mean('data', numpy.array([104, 117, 123]))  # subtract the dataset-mean value in each channel
    caffe_transformer.set_raw_scale('data', 255)  # rescale from [0, 1] to [0, 255]
    caffe_transformer.set_channel_swap('data', (2, 1, 0))  # swap channels from RGB to BGR

    for fulldir, subdir, files in os.walk(INPUT_DIR):
        subdir[:] = [d for d in subdir if d not in EXCLUDE] # Exclude the output directory
        for filename in files:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp', '.gif')):
                try:
                    image_data = open(os.path.join(fulldir, filename), 'rb').read()
                    scores = caffe_preprocess_and_compute(image_data, caffe_transformer=caffe_transformer, caffe_net=nsfw_net, output_layers=['prob'])
                except:
                    continue
                if scores[1] >= SCORE_THRESHOLD:
                    print(os.path.join(fulldir, filename), "NSFW score:" , scores[1])
                    os.rename(os.path.join(fulldir, filename), os.path.join(OUTPUT_DIR, filename))
            
#            else:
#                print(os.path.join(INPUT_DIR, filename), "is not a supported image.")

if __name__ == '__main__':
    main()
