from torchsummary import summary
import numpy as np
import torch.nn as nn
from torchvision import models
import netvlad
import torch

def get_classification_head(backbone_str, concat_vlad = True, concat_csr = False):
        linear = []
        fc = []

        linear.append(nn.Linear(512,512))
        
        if concat_csr:
            fc.append(nn.Linear(1024, 512))
        elif concat_vlad:
            fc.append(nn.Linear(4608, 512))
        
        fc.append(nn.Linear(512,256))
        fc.append(nn.ReLU(inplace=True))
        # fc.append(nn.BatchNorm1d(256))
        fc.append(nn.Linear(256,64))
        fc.append(nn.ReLU(inplace=True))
        # fc.append(nn.BatchNorm1d(64))
        fc.append(nn.Linear(64,2))
        fc.append(nn.Softmax(dim=1))
        
        return nn.Sequential(*linear), nn.Sequential(*fc)

class EmbedModel(nn.Module):
    def __init__(self, netvlad, backbone_str = 'resnet', concat_vlad = True, concat_csr = False, with_attention = True):
        super(EmbedModel, self).__init__()
        self.netvlad = netvlad
        self.backbone_str = backbone_str
        self.concat_vlad = concat_vlad
        self.with_attention = with_attention
        self.concat_csr = concat_csr
        self.selfAttention = nn.MultiheadAttention(4608, num_heads=8, batch_first=True)
        self.crossAttention = nn.MultiheadAttention(4608, num_heads=8, batch_first=True)
        self.linear, self.classification_head = get_classification_head(self.backbone_str, self.concat_vlad, self.concat_csr)
        self.flatten = nn.Flatten()
        self.resnet = models.resnet18(pretrained = True)
        self.feature_extractor = nn.Sequential(*list(self.resnet.children())[:-1])
        for p in self.resnet.parameters():
            p.requires_grad = True
        for p in self.feature_extractor.parameters():
            p.requires_grad = True
        for p in self.netvlad.parameters():
            p.requires_grad = True

    def forward(self, img1, img2):

        out1 = self.feature_extractor(img1)
        out2 = self.feature_extractor(img2)

        out1 = self.flatten(out1)
        out2 = self.flatten(out2)

        out1 = self.linear(out1).unsqueeze(1)
        out2 = self.linear(out2).unsqueeze(1)

        if self.concat_vlad:
            vlad1 = self.netvlad(img1)  ##4096
            vlad2 = self.netvlad(img2)  ##4096

            vlad1 = vlad1.unsqueeze(1)
            vlad2 = vlad2.unsqueeze(1)

            out1 = torch.cat((out1, vlad1), dim = 2)
            out2 = torch.cat((out2, vlad2), dim = 2)

        if self.with_attention:
            # Self attention
            selfout1, _ = self.selfAttention(out1, out1, out1)
            selfout2, _ = self.selfAttention(out2, out2, out2)

            # Cross attention
            out1, _ = self.crossAttention(selfout1, selfout2, selfout2)
            out2, _ = self.crossAttention(selfout2, selfout1, selfout1)
        
        out1 = out1.squeeze(1)
        out2 = out2.squeeze(1)
        joined = torch.pow(out1 - out2, 2)
        out = self.classification_head(joined)
        return out