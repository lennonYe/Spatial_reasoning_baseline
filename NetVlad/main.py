import torch
from torchvision import transforms
from torchsummary import summary
from PIL import Image
import numpy as np
import torch.nn as nn
from torchvision import models
import netvlad
import cv2
from model import EmbedModel
from sklearn.model_selection import train_test_split
from DataLoader import Subset, AVDDataset
from matplotlib import pyplot as plt
from balanced_loss import Loss
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import pandas as pd
import re
import yaml

def calculateIOU(predictions, labels, threshold=0.5):
    """
    Calculate IOU using the predictions and labels
    """
    # Get the predictions above the threshold
    pred = predictions > threshold

    # Calculate the intersection and union
    intersection = np.logical_and(pred, labels)
    union = np.logical_or(pred, labels)
    intersection = np.sum(intersection)
    union = np.sum(union)

    if union == 0:
        return 0.0
    return intersection / union

def getPredictions(model, dataloader, class_balanced=False, concat_csr=False):
    """
    Get the predictions from the model
    """
    model.eval()
    csv_output = []
    pred = np.array([])
    labels = np.array([])
    with torch.no_grad():
        for i, (img1, img2, label) in enumerate(tqdm(dataloader)):
            path = '/'.join([t[0] for t in path])
            img1 = img1.cuda()
            img2 = img2.cuda()
            # Get the predictions
            out = model(img1, img2)[:, 1].cpu().numpy()
            pred = np.append(pred, out)
            labels = np.append(labels, label)
    return pred, labels

def create_labels(scene_names):
    labels = pd.DataFrame()
    for scene_dir in scene_names:
        for floor in os.listdir(scene_dir):
            if os.path.isdir(os.path.join(scene_dir, floor)) and re.search('[0-9]', floor):
                labels_path = os.path.join(scene_dir, floor, 'saved_obs', 'GroundTruth.csv')
                labels_df = pd.read_csv(labels_path)
                labels = pd.concat([labels, labels_df], ignore_index=True)
    return labels

def train(model, iterator, optimizer, criterion):
    epoch_loss = 0

    model.train()
    i = 0

    for (img1, img2, label) in iterator:
        img1 = img1.cuda()
        img2 = img2.cuda()
        label = label.cuda().long()
        optimizer.zero_grad()
        y_pred = model(img1, img2)
        loss = criterion(y_pred, label)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        i+=1
    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion):
    epoch_loss = 0

    y_pred_list = []
    label_list = []

    model.eval()
    with torch.no_grad():

        for (img1, img2, label) in iterator:
            img1 = img1.cuda()
            img2 = img2.cuda()
            label = label.cuda().long()
            y_pred = model(img1, img2)
            y_pred_list.append(y_pred)
            label_list.append(label)
            loss = criterion(y_pred, label)
            epoch_loss += loss.item()

    return epoch_loss / len(iterator), y_pred_list, label_list

def calculateAUC(model, dataloader,predictions,labels, class_balanced=False, concat_csr=False):
    ious = []
    thresholds = np.arange(0, 1.05, 0.05)
    # pred, labels = getPredictions(model, dataloader, class_balanced=class_balanced, concat_csr=concat_csr)
    pred = predictions
    labels = labels
    for t in thresholds:
        iou = calculateIOU(pred, labels, threshold=t)
        ious.append(iou)
    
    print(f'IOUs: {ious}')
    auc = np.trapz(ious, thresholds)
    print(f'AUC: {auc}')
    return auc, ious

def load_config(config_path='config_netvlad.yaml'):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

if __name__ == "__main__":
    # load the best model with PCA (trained by our SFRS)
    vlad = torch.hub.load('yxgeee/OpenIBL', 'vgg16_netvlad', pretrained=True)
    vlad = vlad.cuda()
    config = load_config()
    model = EmbedModel(vlad).cuda()

    for param in vlad.parameters():
        param.requires_grad = True
    # using random seed to divide train and test scenes

    # random.seed(config['random_seed'])
    # all_scenes = glob.glob('./temp/More_vis/*')
    # train_scenes = random.sample(all_scenes, int(len(all_scenes) * 0.8))
    # test_scenes = list(set(all_scenes) - set(train_scenes))

    # or modify the specific desired train scenes and test scenes for consistency
    train_scenes = ['./temp/More_vis/Applewold', './temp/More_vis/Goffs', './temp/More_vis/Mesic', './temp/More_vis/Sanctuary', './temp/More_vis/Silas']
    test_scenes =  ['./temp/More_vis/Anaheim']

    print("Train scenes are:",train_scenes)
    print("Test scenes are",test_scenes)

    labels = create_labels(train_scenes)
    dataset = AVDDataset(
            labels,
        )

    # Split dataset into train and val
    train_idx, val_idx = train_test_split(
            np.arange(len(dataset)), 
            test_size=0.2, 
            random_state=config['random_seed'],
            shuffle=True,
            stratify=dataset.targets
        )

    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)

    train_samples_per_class = train_dataset.class_samples
    val_samples_per_class = val_dataset.class_samples

    train_criterion = Loss(loss_type="binary_cross_entropy", samples_per_class = train_samples_per_class, class_balanced=True)
    val_criterion = Loss(loss_type="binary_cross_entropy", samples_per_class = val_samples_per_class, class_balanced=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    EPOCHS = config['epochs']
    batch_size = config['batch_size']

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    
    train_acc_history = []
    train_loss_history = []
    bestValLoss = float('inf')
    for epoch in range(EPOCHS):
        trainEpochLoss = train(model, train_loader, optimizer, train_criterion)
        valEpochLoss, _, _ = evaluate(model, val_loader, val_criterion)
        if valEpochLoss<bestValLoss:
            bestValLoss = valEpochLoss
            torch.save(model.state_dict(), "best_model_vlad.pt")
        # print("At Epoch: {} Train Loss: {} Train Accuracy: {} Val Loss: {} Val Accuracy: {}".format(epoch, trainEpochLoss, valEpochLoss))
        print("At Epoch: {} Train Loss: {} Val Loss: {}".format(epoch, trainEpochLoss, valEpochLoss))
    print("Best val loss is",bestValLoss)
    model = EmbedModel(vlad).cuda()
    model.load_state_dict(torch.load("best_model_vlad.pt"))
    model.eval()  # set the model to evaluation mode

    ##### Testing #####
    test_labels = create_labels(test_scenes)
    test_dataset = AVDDataset(
            test_labels,
        )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    _, y_pred_list, label_list = evaluate(model, test_loader, val_criterion)
    y_pred_np_list = [tensor.cpu().numpy() for tensor in y_pred_list]
    label_np_list = [tensor.cpu().numpy() for tensor in label_list]
    test_auc, test_iou = calculateAUC(model, test_loader,y_pred_np_list,label_np_list)
    print("Test iou is",test_iou)
    print("The auc is:",test_auc)
