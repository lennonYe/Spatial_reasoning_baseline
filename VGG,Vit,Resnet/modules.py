import os
import math
import json
import torch
import torch.nn as nn
from timm import create_model
import pytorch_lightning as pl
import torchvision.transforms as T
from torchvision import models
import src.dataloaders.augmentations as A
from torchvision.transforms.transforms import ToTensor
from src.models.backbones import FeatureLearner, FeedForward
import itertools
from get_boxes import get_boxes
import timm
import numpy as np
import torch.nn.functional as F
from torch import Tensor
from src.lightning.modules.moco2_module_old import MocoV2
from PIL import Image, ImageDraw

class PositionalEncoding(pl.LightningModule):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        x = x + self.pe[:,:x.size(1)]
        return self.dropout(x)

class AttentionModule(pl.LightningModule):
    def __init__(self, d_model=512, nhead=8, dim_feedforward=2048, dropout=0.1, layer_norm_eps=1e-5, batch_first=True):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)

        self.pos_encoder = PositionalEncoding(d_model, dropout)

        # Feedforward a
        self.linear1a = nn.Linear(d_model, dim_feedforward)
        self.dropouta = nn.Dropout(dropout)
        self.linear2a = nn.Linear(dim_feedforward, d_model)

        # Feedforward b
        self.linear1b = nn.Linear(d_model, dim_feedforward)
        self.dropoutb = nn.Dropout(dropout)
        self.linear2b = nn.Linear(dim_feedforward, d_model)

        # Norms
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm4 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)

        # Activation
        self.activation = nn.ReLU()

    def forward(self, src1: Tensor, src2: Tensor) -> Tensor:
        x1 = self.pos_encoder(src1)
        x2 = self.pos_encoder(src2)

        # Self-attention
        x1 = self.norm1(x1 + self._sa_block(x1))
        x2 = self.norm1(x2 + self._sa_block(x2))

        # Feedforward a
        x1 = self.norm2(x1 + self._ff_blocka(x1))
        x2 = self.norm2(x2 + self._ff_blocka(x2))

        # Cross-attention
        x1_c = self.norm3(x1 + self._ca_block(x1, x2))
        x2_c = self.norm3(x2 + self._ca_block(x2, x1))

        # Feedforward b
        x1 = self.norm4(x1_c + self._ff_blockb(x1_c))
        x2 = self.norm4(x2_c + self._ff_blockb(x2_c))

        return x1, x2
    
    def _sa_block(self, x: Tensor) -> Tensor:
        x = self.self_attn(x, x, x)[0]
        return self.dropout1(x)

    def _ca_block(self, x1: Tensor, x2: Tensor) -> Tensor:
        x = self.cross_attn(x1, x2, x2)[0]
        return self.dropout2(x)
    
    def _ff_blocka(self, x: Tensor) -> Tensor:
        x = self.linear2a(self.dropouta(self.activation(self.linear1a(x))))
        return self.dropout3(x)
    
    def _ff_blockb(self, x: Tensor) -> Tensor:
        x = self.linear2b(self.dropoutb(self.activation(self.linear1b(x))))
        return self.dropout4(x)

class ViTModule(pl.LightningModule):
    def __init__(self, pretrained=True):
        super().__init__()
        self.encoder = create_model('vit_base_patch16_224', pretrained=pretrained)
        # self.encoder.eval()
        # for param in self.encoder.parameters():
        #     param.requires_grad = False

    # def forward_features(self, x):
    #     x = self.path_embed(x)
    #     cls_token = self.cls_token.expand(x.shape[0], -1, -1)
    #     if self.dist_token is None:
    #         x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim = 1)
    #     x = self.pos_drop(x + self.pos_embed)
    #     x = self.blocks(x)
    #     if self.dist_token is None:
    #         return self.pre_logits(x[:, 0])
    #     else:
    #         return x[:, 0], x[:, 1]
       
    def forward(self, img1, img2):
        # self.encoder.eval()       
        img_features = []
        for img in [img1, img2]:
            cls_token = self.encoder.cls_token.expand(img.shape[0], -1, -1)
            feat = self.encoder.patch_embed(img)

            feat = torch.cat((cls_token, feat), dim=1) + self.encoder.pos_embed
            feat = self.encoder.pos_drop(feat)
            
            for block in self.encoder.blocks:
                feat = block(feat)

            feat = self.encoder.norm(feat)
            # features = feat[:,0].squeeze(0).detach()
            features = feat[:,0].squeeze(0)
            img_features.append(features)
        
        return img_features[0], img_features[1] 
       
class VGGModule(pl.LightningModule):
    def __init__(self, pretrained=True):
        super().__init__()
        model_ver = models.vgg16(pretrained=pretrained)
        layers = list(model_ver.children())[:-1]
        self.encoder = nn.Sequential(*layers)
        # self.encoder.eval()
        # for param in self.encoder.parameters():
        #     param.requires_grad = False

    def forward(self, img1, img2):
        features1 = self.encoder(img1)
        features2 = self.encoder(img2)

        return features1, features2
    
class ResNetModule(pl.LightningModule):
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(pretrained=True)
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])
        # self.encoder.eval()
        # for param in self.encoder.parameters():
        #         param.requires_grad = False

    def forward(self,img1,img2):
        features1 = self.encoder(img1)
        features2 = self.encoder(img2)

        return features1, features2

class CSRModule(pl.LightningModule):
    def __init__(self):
        super().__init__()
        
        self.encoder = MocoV2.load_from_checkpoint('checkpoints/csr_scene.ckpt').to('cuda')

        for param in self.encoder.parameters():
                param.requires_grad = False

    def forward(self, img1, img2, imgName1, imgName2, path):
        self.encoder.eval()
        #feature1, feature2 = get_csr_features(self.encoder, img1, img2, imgName1, imgName2, path = None)
        feature1, feature2 = get_csr_features(self.encoder, img1, img2, imgName1, imgName2, path)
        return feature1, feature2


def create_batch(keys, boxes, im):
    mask_1 = torch.zeros((len(keys), 1, 224, 224))
    mask_2 = torch.zeros((len(keys), 1, 224, 224))
    image = torch.zeros((len(keys), 3, 224, 224))
    t = ToTensor()
    tensor_image = t(im)
    for i, k in enumerate(keys):
        mask_1[i] = boxes[k[0]]
        mask_2[i] = boxes[k[1]]
        image[i] = torch.clone(tensor_image)

    return {'mask_1': mask_1, 'mask_2': mask_2, 'image': image}

def get_max_boxes(path):

    filepath = os.path.join(path, 'boxes')
    # print('Determining maximum number of objects')
    # print(f'Path: {filepath}')
    max_num_obj = 0
    for filename in os.listdir(filepath):
        with open(filepath+'/'+filename) as f:
            boxes = json.load(f)
        if len(boxes) > max_num_obj:
            max_num_obj = len(boxes)
    # print(f'Maximum number of objects: {max_num_obj}')
    max_num_obj = max_num_obj ** 2
    return max_num_obj

def get_boxes(corners):
    boxes = {}
    # If there are no detected objects (e.g. 5.jpg in Training set)
    if len(corners) == 0:
        box = Image.new('L', (224, 224))
        tmp = ImageDraw.Draw(box)
        tmp.rectangle([(0, 0), (0, 0)], fill="white")
        trans = T.ToTensor()
        boxes[0] = trans(box)
        return boxes

    for key in corners.keys():
        top = tuple(corners[str(key)]['top'])
        bottom = tuple(corners[str(key)]['bottom'])

        # Convert str in tuples to integers
        top = tuple(map(float, top))
        bottom = tuple(map(float, bottom))

        box = Image.new('L', (224, 224))
        tmp = ImageDraw.Draw(box)
        tmp.rectangle([top, bottom], fill="white")
        trans = T.ToTensor()
        boxes[key] = trans(box)

    return boxes
def get_csr_features(model, img1, img2, imgName1, imgName2, path = None):
    
    assert path != None , "Path can not be none in CSRModule"
    batch_size = img1.shape[0]
    q_features_list = []
    k_features_list = []
    #Generate csr features for batches
    for i in range(batch_size):
        if type(path) == str:
            img1_single = img1
            img2_single = img2
            imgName1_single = ''.join(imgName1)
            imgName2_single = ''.join(imgName2)
            path_single = path
        elif type(path) != str:
            img1_single = img1[i]
            img2_single = img2[i]
            imgName1_single = imgName1[i]
            imgName2_single = imgName2[i]
            path_single = path[i]
        max_num_obj = get_max_boxes(path_single)
        boxes_dir = os.path.join(path_single, 'boxes')

        boxes_file1 = os.path.join(boxes_dir, imgName1_single.split('.', 1)[0]+'.json')
        boxes_file2 = os.path.join(boxes_dir, imgName2_single.split('.', 1)[0]+'.json')
        
        with open(boxes_file1) as f:
            boxes1 = json.load(f)
        
        with open(boxes_file2) as f:
            boxes2 = json.load(f)

        boxes1 = get_boxes(boxes1)
        boxes2 = get_boxes(boxes2)

        edge_pairings1 = list(itertools.permutations(boxes1.keys(), 2))
        edge_pairings2 = list(itertools.permutations(boxes2.keys(), 2))
        self_pairings1 = [(j, j) for j in boxes1]
        self_pairings2 = [(j, j) for j in boxes2]
        
        keys1 = self_pairings1 + edge_pairings1
        keys2 = self_pairings2 + edge_pairings2
        img1_create_batch = Image.open(os.path.join(path_single, imgName1_single))
        img1_create_batch = img1_create_batch.resize((224, 224))
        img2_create_batch = Image.open(os.path.join(path_single, imgName2_single))
        img2_create_batch = img2_create_batch.resize((224, 224))
        x1 = create_batch(keys1, boxes1, img1_create_batch)
        x2 = create_batch(keys2, boxes2, img2_create_batch)

        A.TestTransform(x1)
        A.TestTransform(x2)

        # bx5x224x224
        x_instance1 = torch.cat((x1['image'], x1['mask_1'], x1['mask_2']), 1)
        x_instance2 = torch.cat((x2['image'], x2['mask_1'], x2['mask_2']), 1)

        # Reshape x_instance to be max_num_objx5x224x224 (maximum number of objects detected out of entire set)
        while x_instance1.shape[0] < max_num_obj:
            x_instance1 = torch.cat((x_instance1, x_instance1), dim=0)
        x_instance1 = x_instance1[:max_num_obj]
        x_instance1 = x_instance1.to('cuda')

        while x_instance2.shape[0] < max_num_obj:
            x_instance2 = torch.cat((x_instance2, x_instance2), dim=0)
        x_instance2 = x_instance2[:max_num_obj]
        x_instance2 = x_instance2.to('BCEWithLogitsLoss()(y_hat, label)cuda')

        q_features, k_features = model(x_instance1, x_instance2, update_queue=False)
        q_features = torch.mean(q_features, dim=0)
        k_features = torch.mean(k_features, dim=0)
        q_features_list.append(q_features)
        k_features_list.append(k_features)
    q_features_batch = torch.stack(q_features_list, dim=0)
    k_features_batch = torch.stack(k_features_list, dim=0)

    return q_features_batch, k_features_batch
