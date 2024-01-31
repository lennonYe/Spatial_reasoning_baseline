import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from balanced_loss import Loss
import torchvision.transforms as T
import pytorch_lightning as pl
from NewLoader import AVDDataset
from torch.utils.data import DataLoader, WeightedRandomSampler, Dataset
#from src.lightning.data_modules.contrastive_data_module import \
#    ContrastiveDataModule
from torchvision.transforms.transforms import ToTensor
#from src.lightning.modules.moco2_module_old import MocoV2
#from src.models.backbones import FeatureLearner, FeedForward
#from src.lightning.custom_callbacks import ContrastiveImagePredictionLogger
import os, itertools, json
import torchvision
import torchvision.models as models
import numpy as np
import pandas as pd
from tqdm import tqdm
#import src.dataloaders.augmentations as A
from PIL import Image, ImageDraw
from timm import create_model
import get_boxes
from modules import ViTModule, AttentionModule, VGGModule,ResNetModule,CSRModule
from pytorch_lightning import seed_everything

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
    print('Determining maximum number of objects')
    print(f'Path: {filepath}')
    for filename in os.listdir(filepath):
        with open(filepath+'/'+filename) as f:
            boxes = json.load(f)
        if len(boxes) > max_num_obj:
            max_num_obj = len(boxes)
    print(f'Maximum number of objects: {max_num_obj}')
    max_num_obj = max_num_obj ** 2
    return max_num_obj

def get_features(model_name, model, img, img_name, path = None, encoded_type = None):
    max_num_obj = get_max_boxes(path)
    if model_name == "csr":
        # Load boxes
        boxes_dir = os.path.join(path, 'boxes')
        boxes_file = os.path.join(boxes_dir, img_name.split('.',1)[0]+'.json')
        with open(boxes_file) as f:
            boxes = json.load(f)  
        boxes = get_boxes(boxes)
        edge_pairings = list(itertools.permutations(boxes.keys(), 2))
        self_pairings = [(j, j) for j in boxes]
        keys = self_pairings + edge_pairings
        x = create_batch(keys, boxes, img)
        A.TestTransform(x)
        # bx5x224x224
        x_instance = torch.cat((x['image'], x['mask_1'], x['mask_2']), 1)
        # Reshape x_instance to be max_num_objx5x224x224 (maximum number of objects detected out of entire set)
        while x_instance.shape[0] < max_num_obj:
            x_instance = torch.cat((x_instance,x_instance), dim=0)
        x_instance = x_instance[:max_num_obj]
        x_instance = x_instance.to('cuda')
        q_features, k_features = model(x_instance, x_instance, update_queue=False)

        q_features = torch.mean(q_features, dim=0).squeeze(0).detach().cpu().numpy()
        k_features = torch.mean(k_features, dim=0).squeeze(0).detach().cpu().numpy()
        if encoded_type == "q":
            return q_features
        return k_features
            
class ClassificationModel(pl.LightningModule):
    def __init__(self, train_dataset, val_dataset, lr=1e-4, batch_size=1, backbone_str='vit', class_balanced=False, with_attention=False, concat_csr=False, seed = 0):
        
        super(ClassificationModel, self).__init__()
        self.seed = seed
        seed_everything(self.seed, workers = True)
        self.save_hyperparameters()
        self.lr = lr
        self.batch_size = batch_size
        self.backbone_str = backbone_str
        self.with_attention = with_attention
        self.class_balanced = class_balanced
        self.concat_csr = concat_csr

        # Classification Head
        self.linear, self.classification_head = self.get_classification_head(self.backbone_str, self.concat_csr)
        self.flatten = nn.Flatten()
        self.encoder = self.get_encoder(self.backbone_str)
        # self.encoder.eval()
        self.encoder.train()

        if self.concat_csr:
            self.csr_encoder = self.get_encoder("csr")
            self.csr_encoder.train()

        # Attention
        if self.with_attention:
            if self.concat_csr:
                self.AttentionModule = AttentionModule(
                    d_model=1024,
                    nhead=8,
                    batch_first=True,
                )
            else:
                self.AttentionModule = AttentionModule(
                    d_model=512,
                    nhead=8,
                    batch_first=True,
                )

        # Data Modules
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        # Criterion
        if class_balanced:
            self.train_samples_per_class = self.train_dataset.class_samples
            self.val_samples_per_class = self.val_dataset.class_samples

            self.train_bce_loss = Loss(loss_type="binary_cross_entropy", samples_per_class=self.train_samples_per_class, class_balanced=True)
            self.val_bce_loss = Loss(loss_type="binary_cross_entropy", samples_per_class=self.val_samples_per_class, class_balanced=True)

            # self.train_bce_loss = Loss(loss_type="focal_loss", samples_per_class=self.train_samples_per_class, class_balanced=True)
            # self.val_bce_loss = Loss(loss_type="focal_loss", samples_per_class=self.val_samples_per_class, class_balanced=True)

        else:
            # self.train_bce_loss = nn.BCELoss()
            # self.val_bce_loss = nn.BCELoss()
            self.val_bce_loss = torch.nn.BCEWithLogitsLoss()
            self.train_bce_loss = torch.nn.BCEWithLogitsLoss()
    
    def get_encoder(self, backbone_str):
        if backbone_str == 'vit':
            return ViTModule(pretrained=True)
        elif backbone_str == 'csr':
            return CSRModule()
        elif backbone_str == 'vgg':
            return VGGModule(pretrained=True)
        elif backbone_str == 'resnet':
            return ResNetModule()
        else:
            raise ValueError(f"backbone_str {backbone_str} not supported")

    def get_classification_head(self, backbone_str, concat_csr = False):
        linear = []
        fc = []
        if backbone_str == 'vgg':
            linear.append(nn.Linear(25088,512))
                
        elif backbone_str == 'vit':
            linear.append(nn.Linear(768,512))

        elif backbone_str == 'csr':
            linear.append(nn.Linear(512,512))

        elif backbone_str == 'resnet':
            linear.append(nn.Linear(512,512))
            
        else:
            raise ValueError(f"backbone_str {backbone_str} not supported")
        
        if concat_csr:
            fc.append(nn.Linear(1024, 512))
        else:
            fc.append(nn.Linear(512,512))
        fc.append(nn.ReLU(inplace=True))
        fc.append(nn.BatchNorm1d(512))
        fc.append(nn.Linear(512,256))
        fc.append(nn.ReLU(inplace=True))
        fc.append(nn.BatchNorm1d(256))
        fc.append(nn.Linear(256,2))
        fc.append(nn.Softmax(dim=1))
        return nn.Sequential(*linear), nn.Sequential(*fc)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def forward(self, img1, img2, imageName1=None, imageName2=None, path=None):
        # img1 = img1.type_as(img1, device = self.device)
        # img2 = img2.type_as(img2, device = self.device)
        #input1,input2 = self.encoder(img1,img2)
        if self.backbone_str == 'csr':
            input1,input2 = self.encoder(img1,img2,imageName1,imageName2,path)
        else:
            input1,input2 = self.encoder(img1,img2)
        
        if len(input1.shape) == 1:
            input1 = input1.unsqueeze(0)
            input2 = input2.unsqueeze(0)

        input1 = self.flatten(input1)
        input2 = self.flatten(input2)

        out1 = self.linear(input1)
        out2 = self.linear(input2)

        if self.concat_csr:
            csr1 = self.csr_encoder(img1)
            csr2 = self.csr_encoder(img2)

            csr1 = csr1.unsqueeze(1)
            csr2 = csr2.unsqueeze(1)

            out1 = torch.cat((out1, csr1), dim=2)
            out2 = torch.cat((out2, csr2), dim=2)
        """if self.backbone_str == "csr" or self.concat_csr == True:
            input1 = get_features(self.backbone_str, self.backbone, img1, image_1_name, path,"q")
            input2 = get_features(self.backbone_str, self.backbone, img2, image_2_name, path,"k")
        else:
            input1 = get_features(self.backbone_str, self.backbone, img1, image_1_name)
            input2 = get_features(self.backbone_str, self.backbone, img2, image_2_name)"""

        if self.with_attention:
            src1 = out1.clone()
            src2 = out2.clone()
            out1 = self.AttentionModule(src1, src2)
            out2 = self.AttentionModule(src2, src1)

        #concat = torch.cat((out1, out2), dim=1)
        
        # Join both siamese outputs
        #joined = torch.abs(out1 - out2)
        
        # Try L2 distance
        joined = torch.pow(out1 - out2, 2)

        out = self.classification_head(joined)
        return out

    def training_step(self, batch, batch_idx):
        img1, img2, label, imageName1, imageName2, path = batch

        if self.class_balanced:
            label = label.long()
        else:
            label = label.view(-1,1).float()
        
        if self.concat_csr or self.backbone_str == 'csr':
            y_hat = self(img1, img2, imageName1, imageName2, path)
        else:
            y_hat = self(img1, img2)

        # If not class balanced, only use second class prediction (Confidence in positive pair)
        if not self.class_balanced:
            y_hat = y_hat[:,1].view(-1,1)

        loss = self.train_bce_loss(y_hat, label)
        # loss = torch.nn.BCEWithLogitsLoss()(y_hat, label)


        print("Training Loss: ", loss)
        self.log('train_loss', loss)
        tensorboard_logs = {'train_loss': loss}

        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        img1, img2, label, image_1_name, image_2_name, path = batch

        if self.class_balanced:
            label = label.long()
        else:
            label = label.view(-1,1).float()

        if self.concat_csr or self.backbone_str == "csr":
            y_hat = self(img1, img2, image_1_name, image_2_name, path)
        else:
            y_hat = self(img1, img2)

        # If not class balanced, only use second class prediction (Confidence in positive pair)
        if not self.class_balanced:
            y_hat = y_hat[:,1].view(-1,1)

        loss = self.val_bce_loss(y_hat, label)
        # loss = torch.nn.BCEWithLogitsLoss()(y_hat, label)
        return {'val_loss': loss}
    
    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        self.log('val_loss', avg_loss)
        return {'val_loss': avg_loss, 'log': tensorboard_logs}
    
    def train_dataloader(self):
        train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=8)
        return train_loader
    
    def val_dataloader(self):
        val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=8)
        return val_loader
