import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torchvision.transforms.transforms import ToTensor
import torchvision.models as models
import numpy as np
import pandas as pd
import itertools
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
# from src.lightning.modules.moco2_module_old import MocoV2
# from src.models.backbones import FeatureLearner, FeedForward
# import src.dataloaders.augmentations as A
from PIL import Image, ImageDraw
from timm import create_model

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

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

def get_features(path):
    """
    Given a dataset, extract features from each image using the selected models and save them as tensors.
    """
    
    # Determine the maximum number of objects over all images
    max_num_obj = 0

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

    models_dict = {}
    # Load models
    print('Loading models')
    
    print('Loading vgg')
    model_ver = models.vgg16(pretrained=True).to('cuda')
    layers = list(model_ver.children())[:-1]
    vgg = nn.Sequential(*layers)
    vgg.eval()
    for param in vgg.parameters():
        param.requires_grad = False

    models_dict['vgg'] = vgg

    print('Loading vit')
    vit = create_model('vit_base_patch16_224', pretrained=True).to('cuda')
    vit.eval()
    for param in vit.parameters():
        param.requires_grad = False

    models_dict['vit'] = vit

    print('Loading csr')
    csr = MocoV2.load_from_checkpoint('checkpoints/csr_scene.ckpt').to('cuda')
    csr.eval()
    for param in csr.parameters():
        param.requires_grad = False
    
    models_dict['csr'] = csr

    print('Loading resnet')
    resnet = FeatureLearner(
        in_channels=3,
        channel_width=64,
        pretrained=True,
        num_classes=0,
        backbone_str='resnet18').to('cuda')
    resnet.eval()
    for param in resnet.parameters():
        param.requires_grad = False

    models_dict['resnet'] = resnet
    
    # Load dataset
    print('Loading dataset')
    img_dir = os.path.join(path, 'images')

    transforms = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])])

    # Create directories for each model
    for model_name in models_dict.keys():
        print(f'Creating directory for {model_name}')
        # Check if it exists
        if not os.path.exists(os.path.join(path, model_name)):
            os.mkdir(os.path.join(path, model_name))
    
    # Extract features
    with torch.no_grad():
        for model_name, model in models_dict.items():
            print(f'Extracting features using {model_name}')
            if model_name == 'csr':
                # Load boxes
                boxes_dir = os.path.join(path, 'boxes')
                for img_name in tqdm(os.listdir(img_dir)):
                    img = Image.open(os.path.join(img_dir, img_name))
                    img = img.resize((224, 224))

                    # Load boxes
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
                    x_instance = torch.cat((x['image'], x['mask_1'], x['mask_2']),
                                        1)

                    # Reshape x_instance to be max_num_objx5x224x224 (maximum number of objects detected out of entire set)
                    while x_instance.shape[0] < max_num_obj:
                        x_instance = torch.cat((x_instance,x_instance), dim=0)
                    x_instance = x_instance[:max_num_obj]
                    x_instance = x_instance.to('cuda')

                    q_features, k_features = model(x_instance, x_instance, update_queue=False)
                    q_features = torch.mean(q_features, dim=0).squeeze(0).detach().cpu().numpy()
                    k_features = torch.mean(k_features, dim=0).squeeze(0).detach().cpu().numpy()

                    np.save(os.path.join(path, model_name, img_name.split('.',1)[0])+'_q', q_features)
                    np.save(os.path.join(path, model_name, img_name.split('.',1)[0]+'_k'), k_features)
            elif model_name == 'vit':
                for img_name in tqdm(os.listdir(img_dir)):
                    img = Image.open(os.path.join(img_dir, img_name))
                    img = transforms(img).unsqueeze(0).to('cuda')
                    cls_token = model.cls_token.expand(1, -1, -1)

                    feat = model.patch_embed(img)

                    feat = torch.cat((cls_token, feat), dim=1) + model.pos_embed
                    feat = model.pos_drop(feat)
                    
                    for block in model.blocks:
                        feat = block(feat)

                    feat = model.norm(feat)
                    features = feat[:,0].squeeze(0).detach().cpu().numpy() # Just the cls token
                    np.save(os.path.join(path, model_name, img_name.split('.',1)[0]), features)
            else:
                for img_name in tqdm(os.listdir(img_dir)):
                    img = Image.open(os.path.join(img_dir, img_name))
                    img = transforms(img).unsqueeze(0).to('cuda')
                    features = model(img).squeeze(0).detach().cpu().numpy()
                    np.save(os.path.join(path, model_name, img_name.split('.',1)[0]), features)

    return True
