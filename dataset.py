from torchvision.datasets import VisionDataset

from util import BezierSampler, get_valid_points
import os
import json
from PIL import Image
import torch
import numpy as np


class BezierDataset(VisionDataset):
    def __init__(self, root: str, image_set='train', order=3, sample_points=100, transforms=None, transform=None, target_transform=None) -> None:
        super().__init__(root, transforms, transform, target_transform)
        self.degree = order 
        self.sample_points = sample_points
        self.transforms = transforms
        self.root = root
        self.image_set = image_set
        self.bezier_sampler = BezierSampler()

        # Read in data from folder 
        if image_set == 'test':
            self.bezier_points = {}
            with open(os.path.join(root, 'test_label.json'), 'r') as f:
                for line in f:
                    data = json.loads(line)
                    self.bezier_points[data['raw_file']] = {}
                    self.bezier_points[data['raw_file']]['h_samples'] = data['h_samples']
                    self.bezier_points[data['raw_file']]['lanes'] = data['lanes']
        else:
             with open(os.path.join(root, 'bezier_convert.json'), 'r') as f:
                self.bezier_points = json.load(f)
            

        with open(os.path.join(root, f"{image_set}.txt"), "r") as f:
            contents = [x.strip().split(' ')[:2] for x in f.readlines()]

        self.seg_image = {k[1:]: v for k, v in contents}
        self.image_files = list(self.seg_image.keys())

    def __getitem__(self, index: int):
        image_file = self.image_files[index]
        # Get image
        img = Image.open(os.path.join(self.root, image_file))

        if self.image_set == 'test':
            targets = ''
            img, targets = self.transforms(img, targets)
            return img, self.bezier_points[image_file]

        labels = {}
        
        labels['keypoints'] = np.array(self.bezier_points[image_file])
        labels['segmentation_mask'] = Image.open(os.path.join(self.root + self.seg_image[image_file]))

        img, labels = self.transforms(img, labels)

        # Sample points
        if len(labels['keypoints']) == 0:
            labels['sampled_points'] = torch.tensor([], dtype=labels['keypoints'].dtype)
        else:
            labels['sampled_points'] = self.bezier_sampler.sample_points(labels['keypoints'])

        valid_lanes = get_valid_points(labels['sampled_points']).sum(dim=-1) >= 2
        labels['keypoints'] = labels['keypoints'][valid_lanes]
        labels['sample_points'] = labels['sampled_points'][valid_lanes]

        # Map to binary (0 1 255)
        positive_mask = (labels['segmentation_mask'] > 0) * (labels['segmentation_mask'] != 255)
        labels['segmentation_mask'][positive_mask] = 1
        
        return img, labels

    def __len__(self):
        if self.image_set == 'test':
            return len(self.bezier_points)

        return len(self.image_files)

