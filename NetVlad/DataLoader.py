import itertools
import json
import time
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.optimize import linear_sum_assignment
from PIL import Image, ImageDraw
import torchvision.transforms as T
import cv2
from torch.utils.data import Dataset

class Subset(Dataset):
    r"""
    Subset of a dataset at specified indices.

    Args:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """

    def __init__(self, dataset: Dataset, indices) -> None:
        self.dataset = dataset
        self.indices = indices

        # Get class weights and class samples
        self.class_weights, self.class_samples = self._compute_class_weights()
    
    def _compute_class_weights(self):
        # Compute class weights considering only the subset indices
        labels = self.dataset.labels['label'].tolist()
        class_sample_count = [len([idx for idx in self.indices if labels[idx] == t]) for t in set(labels)]
        weight = 1. / torch.Tensor(class_sample_count)
        return weight, class_sample_count

    def __getitem__(self, idx):
        if isinstance(idx, list):
            return self.dataset[[self.indices[i] for i in idx]]
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)

class AVDDataset():
    def __init__(
        self,
        labels
        ):

        # Create DataFrame from csv
        self.labels = labels

        # Targets
        self.targets = self.labels['label'].tolist()

        # Class weights
        self.class_weights, self.class_samples = self._compute_class_weights()

    def _compute_class_weights(self):
        labels = self.labels['label'].tolist()
        class_sample_count = [len([idx for idx in range(len(labels)) if labels[idx] == t]) for t in set(labels)]
        weight = 1. / torch.Tensor(class_sample_count)
        return weight, class_sample_count

    def __getitem__(self, index):
        # Get image names
        p1 = self.labels.iloc[index]['image_1']
        p2 = self.labels.iloc[index]['image_2']

        # read image
        img1 = np.array(Image.open(p1)) # modify the image path according to your need
        img1 = cv2.resize(img1, (224, 224), interpolation = cv2.INTER_LINEAR)
        img2 = np.array(Image.open(p2)) # modify the image path according to your need
        img2 = cv2.resize(img2, (224, 224), interpolation = cv2.INTER_LINEAR)

        img1 = torch.tensor(img1)
        img2 = torch.tensor(img2)

        img1 = img1.cuda()
        img1 = img1/255.0
        img2 = img2.cuda()
        img2 = img2/255.0

        # Get label
        label = torch.tensor(self.labels.iloc[index]['label'])
        
        return img1.permute(2, 1, 0), img2.permute(2, 1, 0), label.float()
    
    def __len__(self):
        return len(self.labels)