import itertools
import json
import time
import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.optimize import linear_sum_assignment

from torchvision.transforms.transforms import ToTensor
from PIL import Image, ImageDraw
import torchvision.transforms as T
from torch.utils.data import Dataset

class VisAVDDataset(Dataset):
    def __init__(self, labels, backbone_str):
        self.labels = labels
        self.backbone_str = backbone_str

        # Targets
        self.targets = self.labels['label'].tolist()

        # Class weights
        self.class_weights, self.class_samples = self._compute_class_weights()
        
    def _compute_class_weights(self):
        #labels = self.labels['label'].tolist()
        class_sample_count = [len([idx for idx in range(len(self.targets)) if self.targets[idx] == t]) for t in set(self.targets)]
        weight = 1. / torch.Tensor(class_sample_count)
        return weight, class_sample_count
    
    def __getitem__(self, index):
        # Get image names
        p1 = self.labels.iloc[index]['image_1']
        p2 = self.labels.iloc[index]['image_2']

        # Get path to images
        saved_obs_path = p1.split('/')[:-1]
        imageName1 = p1.split('/')[-1]
        imageName2 = p2.split('/')[-1]

        img1 = np.array(Image.open(p1))
        img1 = cv2.resize(img1,(224,224),interpolation = cv2.INTER_LINEAR)
        img2 = np.array(Image.open(p2))
        img2 = cv2.resize(img2,(224,224),interpolation = cv2.INTER_LINEAR)

        # Convert to tensors
        img1 = torch.from_numpy(img1)
        img2 = torch.from_numpy(img2)
        
        img1 = img1/255.0
        img2 = img2/255.0
        label = torch.tensor(self.labels.iloc[index]["label"])
        return img1.permute(2, 1, 0), img2.permute(2, 1, 0), label, imageName1, imageName2, saved_obs_path
    
    def __len__(self):
        return len(self.labels)