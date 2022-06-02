import numpy as np

import json
from glob import glob
import os
from scipy.special import comb
from tqdm import tqdm

from util import BezierCurve

IMAGE_HEIGHT=720
IMAGE_WIDTH=1280


def poly_fit():
    pass 


def read_labels(folder):
    labels = []
    for file in glob(os.path.join(folder, '*.json')):
        with open(file, 'r') as f:
            labels += [json.loads(x.strip()) for x in f.readlines()]

    return labels

def extract_coord(data):
    coords = {}
    curve = BezierCurve(4)
    for image in tqdm(data, total=len(data), desc='Perform Bezier fitting'):
        points = []
        for lane in image['lanes']:
            lane_point = []
            x = np.array(lane)
            y = np.array(image['h_samples'])[x != -2]
            if len(y) == 0:
                continue
            x = x[x != -2]
            xpts, ypts = curve.bezier_fit(x, y)
            for x, y in zip(xpts, ypts):
                lane_point.append([round(x, 3), round(y, 3)])
            
            points.append(lane_point)

        coords[image['raw_file']] = points
    
    return coords

def convert_label():
    train_labels = read_labels('data/train_set')
    train_coords = extract_coord(train_labels)

    with open('data/train_set/bezier_convert.json', 'w+') as f:
        json.dump(train_coords, f)

if __name__ == '__main__':
    convert_label()