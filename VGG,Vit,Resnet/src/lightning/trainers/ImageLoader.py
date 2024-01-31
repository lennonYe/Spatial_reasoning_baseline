import itertools
import json
import time
import os

import numpy as np
import src.dataloaders.augmentations as A
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.optimize import linear_sum_assignment
from src.lightning.modules import moco2_module_old
from src.lightning.modules import moco2_module
from src.shared.constants import IMAGE_SIZE
from src.shared.utils import (check_none_or_empty, get_device,
                              load_lightning_inference, render_adj_matrix)
from src.simulation.module_box import GtBoxModule
from src.simulation.state import State
from src.simulation.utils import get_openable_objects, get_pickupable_objects
from torchvision.transforms.transforms import ToTensor

class TestDataset():
    def __init__(self,img_dir,boxes_dir,transform=None):
        self.img_dir = img_dir    
        self.boxes_dir = boxes_dir
        self.transform = transform
    
    def get_boxes(corners):
        boxes = {}
        for key in corners.keys():
            top = corners[str(ind)]['top']
            bottom = corners[str(ind)]['bottom']

            box = Image.new('L', (IMAGE_SIZE, IMAGE_SIZE))
            tmp = ImageDraw.Draw(box)
            tmp.rectangle([top, bottom], fill="white")
            trans = T.ToTensor()
            boxes[key] = trans(box)

        return boxes

    def create_batch(self, keys, boxes, im):
        mask_1 = torch.zeros((len(keys), 1, IMAGE_SIZE, IMAGE_SIZE))
        mask_2 = torch.zeros((len(keys), 1, IMAGE_SIZE, IMAGE_SIZE))
        image = torch.zeros((len(keys), 3, IMAGE_SIZE, IMAGE_SIZE))
        t = ToTensor()
        tensor_image = t(im)
        for i, k in enumerate(keys):
            mask_1[i] = boxes[k[0]]
            mask_2[i] = boxes[k[1]]
            image[i] = torch.clone(tensor_image)

        return {'mask_1': mask_1, 'mask_2': mask_2, 'image': image}

    def __getitem__(self,index):
        # Load image
        image = os.path.join(self.img_dir, str(index + 1) + '.jpg')
        image = Image.open(image)

        # Load boxes
        boxes = os.path.join(self.boxes_dir, str(index + 1) + '.json')
        with open(boxes) as f:
            boxes = json.load(f)
        boxes = self.get_boxes(boxes)

        # Create batch
        edge_pairings = list(itertools.permutations(boxes.keys(), 2))
        self_pairings = [(i, i) for i in boxes]
        keys = self_pairings + edge_pairings

        x = self.create_batch(keys, boxes, image)

        A.TestTransform(x)
        # bx5x224x224
        x_instance = torch.cat((x['image'], x['mask_1'], x['mask_2']),
                            1).to(self.device)
        
        # x_instance goes into model
        return x_instance
        
    def __len__(self):
        # Return number of images in directory
        return len(os.listdir(self.img_dir))