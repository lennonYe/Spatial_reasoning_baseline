import torch
from torchvision import transforms
# from torchsummary import summary
from PIL import Image
import numpy as np
import torch.nn as nn
from torchvision import models
import netvlad
import cv2
from model import EmbedModel
# from sklearn.model_selection import train_test_split
from DataLoader import Subset, AVDDataset
from matplotlib import pyplot as plt
from balanced_loss import Loss
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import pandas as pd
import re
from visLoader import VisAVDDataset

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
        for i, (img1, img2, label, imgName1, imgName2, path) in enumerate(tqdm(dataloader)):
            path = '/'.join([t[0] for t in path])
            img1 = img1.cuda()
            img2 = img2.cuda()
            # Get the predictions
            if concat_csr:
                csr1 = csr1.cuda()
                csr2 = csr2.cuda()
                out = model(img1, img2, imgName1, imgName2, path)[:, 1].cpu().numpy()
                pred = np.append(pred, out)
            else:
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
    epoch_acc = 0

    model.train()
    i = 0

    for (img1, img2, label) in iterator:

        img1 = img1.cuda()
        img2 = img2.cuda()
        label = label.cuda().long()
        
        optimizer.zero_grad()

        y_pred = model(img1, img2)

        # plt.subplot(1, 2, 1)
        # plt.title("Label: {} with predicted: {}".format(label.cpu().numpy(), y_pred.detach().cpu().numpy()))
        # plt.imshow(img1.squeeze(0).permute(2, 1, 0).cpu().numpy())
        # plt.subplot(1, 2, 2)
        # plt.imshow(img2.squeeze(0).permute(2, 1, 0).cpu().numpy())
        # plt.savefig("./TrainedImages/sample_{}.jpg".format(i))
        # plt.close()
        
        loss = criterion(y_pred, label)

        # acc = calculate_accuracy(y_pred, label)

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        # epoch_acc += acc.item()
        i+=1
    return epoch_loss / len(iterator)

def evaluate(model, iterator):
 
    epoch_loss = 0
    epoch_acc = 0

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
            
            # loss = criterion(y_pred, label)
            # acc = calculate_accuracy(y_pred, label)
            # epoch_loss += loss.item()
            # epoch_acc += acc.item()

    return y_pred_list, label_list

def calculateAUC(model, dataloader,predictions=None,labels=None):
 
    ious = []
    thresholds = np.arange(0, 1.05, 0.05)
    pred, labels = getPredictions(model, dataloader)
    
    for t in thresholds:
        iou = calculateIOU(pred, labels, threshold=t)
        ious.append(iou)
    
    print(f'IOUs: {ious}')
    auc = np.trapz(ious, thresholds)
    print(f'AUC: {auc}')
    return auc, ious

if __name__ == "__main__":
    # load the best model with PCA (trained by our SFRS)
    vlad = torch.hub.load('yxgeee/OpenIBL', 'vgg16_netvlad', pretrained=True)

    # # use GPU (optional)
    # vlad = vlad.cuda()

    # model = EmbedModel(vlad).cuda()

    # for param in vlad.parameters():
    #     param.requires_grad = True

    # main_dir = './datasets/realworld/training'
    # label_dir = './datasets/realworld/training/training_csv.csv'
    # test_dir = './datasets/realworld/testing'
    # test_label_dir = './datasets/realworld/testing/testing_csv.csv'
    # # Prepare datasets
    # dataset = AVDDataset(
    #     main_dir, 
    #     label_dir
    # )
    # Split dataset into train and val
    # train_idx, val_idx = train_test_split(
    #     np.arange(len(dataset)), 
    #     test_size=0.2, 
    #     random_state=42,
    #     shuffle=True,
    #     stratify=dataset.targets
    # )
    # train_scenes = ['./temp/More_vis/Spencerville-1', './temp/More_vis/Spencerville-2', './temp/More_vis/Spencerville-5', './temp/More_vis/Spencerville-4', './temp/More_vis/Spencerville-3', './temp/More_vis/Andover-1', './temp/More_vis/Andover-4', './temp/More_vis/Andover-3', './temp/More_vis/Andover-2', './temp/More_vis/Andover-5', './temp/More_vis/Bowlus-4', './temp/More_vis/Bowlus-3', './temp/More_vis/Bowlus-2', './temp/More_vis/Bowlus-5', './temp/More_vis/Bowlus-1', './temp/More_vis/Woonsocket-2', './temp/More_vis/Woonsocket-5', './temp/More_vis/Woonsocket-4', './temp/More_vis/Woonsocket-3', './temp/More_vis/Woonsocket-1', './temp/More_vis/Eastville-1', './temp/More_vis/Eastville-5', './temp/More_vis/Eastville-2', './temp/More_vis/Eastville-3', './temp/More_vis/Eastville-4', './temp/More_vis/Nuevo-1', './temp/More_vis/Nuevo-3', './temp/More_vis/Nuevo-4', './temp/More_vis/Nuevo-5', './temp/More_vis/Nuevo-2', './temp/More_vis/Hominy-4', './temp/More_vis/Hominy-3', './temp/More_vis/Hominy-2', './temp/More_vis/Hominy-5', './temp/More_vis/Hominy-1', './temp/More_vis/Sumas-5', './temp/More_vis/Sumas-2', './temp/More_vis/Sumas-3', './temp/More_vis/Sumas-4', './temp/More_vis/Sumas-1', './temp/More_vis/Sasakwa-1', './temp/More_vis/Sasakwa-4', './temp/More_vis/Sasakwa-3', './temp/More_vis/Sasakwa-2', './temp/More_vis/Sasakwa-5', './temp/More_vis/Seward-1', './temp/More_vis/Seward-2', './temp/More_vis/Seward-5', './temp/More_vis/Seward-4', './temp/More_vis/Seward-3', './temp/More_vis/Dunmor-3', './temp/More_vis/Dunmor-4', './temp/More_vis/Dunmor-5', './temp/More_vis/Dunmor-2', './temp/More_vis/Dunmor-1', './temp/More_vis/Rosser-4', './temp/More_vis/Rosser-3', './temp/More_vis/Rosser-2', './temp/More_vis/Rosser-5', './temp/More_vis/Rosser-1', './temp/More_vis/Ballou-2', './temp/More_vis/Ballou-5', './temp/More_vis/Ballou-4', './temp/More_vis/Ballou-3', './temp/More_vis/Ballou-1', './temp/More_vis/Sodaville-4', './temp/More_vis/Sodaville-3', './temp/More_vis/Sodaville-2', './temp/More_vis/Sodaville-5', './temp/More_vis/Sodaville-1', './temp/More_vis/Convoy-1', './temp/More_vis/Convoy-4', './temp/More_vis/Convoy-3', './temp/More_vis/Convoy-2', './temp/More_vis/Convoy-5', './temp/More_vis/Adrian-1', './temp/More_vis/Adrian-4', './temp/More_vis/Adrian-3', './temp/More_vis/Adrian-2', './temp/More_vis/Adrian-5', './temp/More_vis/Elmira-1', './temp/More_vis/Elmira-4', './temp/More_vis/Elmira-3', './temp/More_vis/Elmira-2', './temp/More_vis/Elmira-5', './temp/More_vis/Ribera-5', './temp/More_vis/Ribera-2', './temp/More_vis/Ribera-3', './temp/More_vis/Ribera-4', './temp/More_vis/Ribera-1', './temp/More_vis/Avonia-1', './temp/More_vis/Avonia-2', './temp/More_vis/Avonia-5', './temp/More_vis/Avonia-4', './temp/More_vis/Avonia-3', './temp/More_vis/Bolton-2', './temp/More_vis/Bolton-5', './temp/More_vis/Bolton-4', './temp/More_vis/Bolton-3', './temp/More_vis/Bolton-1', './temp/More_vis/Scioto-3', './temp/More_vis/Scioto-4', './temp/More_vis/Scioto-5', './temp/More_vis/Scioto-2', './temp/More_vis/Scioto-1', './temp/More_vis/Kerrtown-5', './temp/More_vis/Kerrtown-2', './temp/More_vis/Kerrtown-3', './temp/More_vis/Kerrtown-4', './temp/More_vis/Kerrtown-1', './temp/More_vis/Roxboro-1', './temp/More_vis/Roxboro-5', './temp/More_vis/Roxboro-2', './temp/More_vis/Roxboro-3', './temp/More_vis/Roxboro-4', './temp/More_vis/Roeville-3', './temp/More_vis/Roeville-4', './temp/More_vis/Roeville-5', './temp/More_vis/Roeville-2', './temp/More_vis/Roeville-1', './temp/More_vis/Sands-5', './temp/More_vis/Sands-2', './temp/More_vis/Sands-3', './temp/More_vis/Sands-4', './temp/More_vis/Sands-1', './temp/More_vis/Parole-4', './temp/More_vis/Parole-3', './temp/More_vis/Parole-2', './temp/More_vis/Parole-5', './temp/More_vis/Parole-1', './temp/More_vis/Greigsville-1', './temp/More_vis/Greigsville-5', './temp/More_vis/Greigsville-2', './temp/More_vis/Greigsville-3', './temp/More_vis/Greigsville-4', './temp/More_vis/Eagerville-1', './temp/More_vis/Eagerville-3', './temp/More_vis/Eagerville-4', './temp/More_vis/Eagerville-5', './temp/More_vis/Eagerville-2', './temp/More_vis/Spotswood-1', './temp/More_vis/Spotswood-2', './temp/More_vis/Spotswood-5', './temp/More_vis/Spotswood-4', './temp/More_vis/Spotswood-3', './temp/More_vis/Albertville-3', './temp/More_vis/Albertville-4', './temp/More_vis/Albertville-5', './temp/More_vis/Albertville-2', './temp/More_vis/Albertville-1', './temp/More_vis/Brevort-1', './temp/More_vis/Brevort-3', './temp/More_vis/Brevort-4', './temp/More_vis/Brevort-5', './temp/More_vis/Brevort-2', './temp/More_vis/Roane-3', './temp/More_vis/Roane-4', './temp/More_vis/Roane-5', './temp/More_vis/Roane-2', './temp/More_vis/Roane-1', './temp/More_vis/Stokes-3', './temp/More_vis/Stokes-4', './temp/More_vis/Stokes-5', './temp/More_vis/Stokes-2', './temp/More_vis/Stokes-1', './temp/More_vis/Goffs-3', './temp/More_vis/Goffs-4', './temp/More_vis/Goffs-5', './temp/More_vis/Goffs-2', './temp/More_vis/Goffs-1', './temp/More_vis/Nimmons-1', './temp/More_vis/Nimmons-2', './temp/More_vis/Nimmons-5', './temp/More_vis/Nimmons-4', './temp/More_vis/Nimmons-3', './temp/More_vis/Sanctuary-2', './temp/More_vis/Sanctuary-5', './temp/More_vis/Sanctuary-4', './temp/More_vis/Sanctuary-3', './temp/More_vis/Sanctuary-1', './temp/More_vis/Delton-1', './temp/More_vis/Delton-2', './temp/More_vis/Delton-5', './temp/More_vis/Delton-4', './temp/More_vis/Delton-3', './temp/More_vis/Stanleyville-1', './temp/More_vis/Stanleyville-3', './temp/More_vis/Stanleyville-4', './temp/More_vis/Stanleyville-5', './temp/More_vis/Stanleyville-2', './temp/More_vis/Haxtun-3', './temp/More_vis/Haxtun-4', './temp/More_vis/Haxtun-5', './temp/More_vis/Haxtun-2', './temp/More_vis/Haxtun-1', './temp/More_vis/Nicut-1', './temp/More_vis/Nicut-5', './temp/More_vis/Nicut-2', './temp/More_vis/Nicut-3', './temp/More_vis/Nicut-4', './temp/More_vis/Maryhill-1', './temp/More_vis/Maryhill-4', './temp/More_vis/Maryhill-3', './temp/More_vis/Maryhill-2', './temp/More_vis/Maryhill-5', './temp/More_vis/Rancocas-1', './temp/More_vis/Rancocas-2', './temp/More_vis/Rancocas-5', './temp/More_vis/Rancocas-4', './temp/More_vis/Rancocas-3', './temp/More_vis/Micanopy-4', './temp/More_vis/Micanopy-3', './temp/More_vis/Micanopy-2', './temp/More_vis/Micanopy-5', './temp/More_vis/Micanopy-1', './temp/More_vis/Capistrano-5', './temp/More_vis/Capistrano-2', './temp/More_vis/Capistrano-3', './temp/More_vis/Capistrano-4', './temp/More_vis/Capistrano-1', './temp/More_vis/Hainesburg-5', './temp/More_vis/Hainesburg-2', './temp/More_vis/Hainesburg-3', './temp/More_vis/Hainesburg-4', './temp/More_vis/Hainesburg-1', './temp/More_vis/Reyno-1', './temp/More_vis/Reyno-3', './temp/More_vis/Reyno-4', './temp/More_vis/Reyno-5', './temp/More_vis/Reyno-2', './temp/More_vis/Azusa-1', './temp/More_vis/Azusa-4', './temp/More_vis/Azusa-3', './temp/More_vis/Azusa-2', './temp/More_vis/Azusa-5', './temp/More_vis/Colebrook-4', './temp/More_vis/Colebrook-3', './temp/More_vis/Colebrook-2', './temp/More_vis/Colebrook-5', './temp/More_vis/Colebrook-1', './temp/More_vis/Oyens-1', './temp/More_vis/Oyens-2', './temp/More_vis/Oyens-5', './temp/More_vis/Oyens-4', './temp/More_vis/Oyens-3', './temp/More_vis/Mesic-3', './temp/More_vis/Mesic-4', './temp/More_vis/Mesic-5', './temp/More_vis/Mesic-2', './temp/More_vis/Mesic-1', './temp/More_vis/Pleasant-1', './temp/More_vis/Pleasant-2', './temp/More_vis/Pleasant-5', './temp/More_vis/Pleasant-4', './temp/More_vis/Pleasant-3', './temp/More_vis/Arkansaw-1', './temp/More_vis/Arkansaw-2', './temp/More_vis/Arkansaw-5', './temp/More_vis/Arkansaw-4', './temp/More_vis/Arkansaw-3', './temp/More_vis/Sawpit-1', './temp/More_vis/Sawpit-4', './temp/More_vis/Sawpit-3', './temp/More_vis/Sawpit-2', './temp/More_vis/Sawpit-5', './temp/More_vis/Anaheim-3', './temp/More_vis/Anaheim-4', './temp/More_vis/Anaheim-5', './temp/More_vis/Anaheim-2', './temp/More_vis/Anaheim-1', './temp/More_vis/Quantico-1', './temp/More_vis/Quantico-2', './temp/More_vis/Quantico-5', './temp/More_vis/Quantico-4', './temp/More_vis/Quantico-3', './temp/More_vis/Hometown-1', './temp/More_vis/Hometown-5', './temp/More_vis/Hometown-2', './temp/More_vis/Hometown-3', './temp/More_vis/Hometown-4', './temp/More_vis/Angiola-5', './temp/More_vis/Angiola-2', './temp/More_vis/Angiola-3', './temp/More_vis/Angiola-4', './temp/More_vis/Angiola-1', './temp/More_vis/Eudora-3', './temp/More_vis/Eudora-4', './temp/More_vis/Eudora-5', './temp/More_vis/Eudora-2', './temp/More_vis/Eudora-1', './temp/More_vis/Mosquito-1', './temp/More_vis/Mosquito-5', './temp/More_vis/Mosquito-2', './temp/More_vis/Mosquito-3', './temp/More_vis/Mosquito-4', './temp/More_vis/Pablo-4', './temp/More_vis/Pablo-3', './temp/More_vis/Pablo-2', './temp/More_vis/Pablo-5', './temp/More_vis/Pablo-1', './temp/More_vis/Placida-2', './temp/More_vis/Placida-5', './temp/More_vis/Placida-4', './temp/More_vis/Placida-3', './temp/More_vis/Placida-1', './temp/More_vis/Hillsdale-1', './temp/More_vis/Hillsdale-5', './temp/More_vis/Hillsdale-2', './temp/More_vis/Hillsdale-3', './temp/More_vis/Hillsdale-4', './temp/More_vis/Stilwell-3', './temp/More_vis/Stilwell-4', './temp/More_vis/Stilwell-5', './temp/More_vis/Stilwell-2', './temp/More_vis/Stilwell-1', './temp/More_vis/Mobridge-5', './temp/More_vis/Mobridge-2', './temp/More_vis/Mobridge-3', './temp/More_vis/Mobridge-4', './temp/More_vis/Mobridge-1', './temp/More_vis/Sisters-3', './temp/More_vis/Sisters-4', './temp/More_vis/Sisters-5', './temp/More_vis/Sisters-2', './temp/More_vis/Sisters-1', './temp/More_vis/Swormville-4', './temp/More_vis/Swormville-3', './temp/More_vis/Swormville-2', './temp/More_vis/Swormville-5', './temp/More_vis/Swormville-1', './temp/More_vis/Cantwell-5', './temp/More_vis/Cantwell-2', './temp/More_vis/Cantwell-3', './temp/More_vis/Cantwell-4', './temp/More_vis/Cantwell-1', './temp/More_vis/Superior-2', './temp/More_vis/Superior-5', './temp/More_vis/Superior-4', './temp/More_vis/Superior-3', './temp/More_vis/Superior-1']
    # test_scenes = ['./temp/More_vis/Hambleton-4', './temp/More_vis/Hambleton-3', './temp/More_vis/Hambleton-2', './temp/More_vis/Hambleton-5', './temp/More_vis/Hambleton-1', './temp/More_vis/Monson-5', './temp/More_vis/Monson-2', './temp/More_vis/Monson-3', './temp/More_vis/Monson-4', './temp/More_vis/Monson-1', './temp/More_vis/Mosinee-3', './temp/More_vis/Mosinee-4', './temp/More_vis/Mosinee-5', './temp/More_vis/Mosinee-2', './temp/More_vis/Mosinee-1', './temp/More_vis/Crandon-1', './temp/More_vis/Crandon-5', './temp/More_vis/Crandon-2', './temp/More_vis/Crandon-3', './temp/More_vis/Crandon-4', './temp/More_vis/Dryville-1', './temp/More_vis/Dryville-5', './temp/More_vis/Dryville-2', './temp/More_vis/Dryville-3', './temp/More_vis/Dryville-4', './temp/More_vis/Nemacolin-1', './temp/More_vis/Nemacolin-4', './temp/More_vis/Nemacolin-3', './temp/More_vis/Nemacolin-2', './temp/More_vis/Nemacolin-5', './temp/More_vis/Shelbiana-3', './temp/More_vis/Shelbiana-4', './temp/More_vis/Shelbiana-5', './temp/More_vis/Shelbiana-2', './temp/More_vis/Shelbiana-1', './temp/More_vis/Annawan-1', './temp/More_vis/Annawan-4', './temp/More_vis/Annawan-3', './temp/More_vis/Annawan-2', './temp/More_vis/Annawan-5', './temp/More_vis/Denmark-3', './temp/More_vis/Denmark-4', './temp/More_vis/Denmark-5', './temp/More_vis/Denmark-2', './temp/More_vis/Denmark-1', './temp/More_vis/Edgemere-1', './temp/More_vis/Edgemere-3', './temp/More_vis/Edgemere-4', './temp/More_vis/Edgemere-5', './temp/More_vis/Edgemere-2', './temp/More_vis/Silas-1', './temp/More_vis/Silas-4', './temp/More_vis/Silas-3', './temp/More_vis/Silas-2', './temp/More_vis/Silas-5', './temp/More_vis/Soldier-1', './temp/More_vis/Soldier-2', './temp/More_vis/Soldier-5', './temp/More_vis/Soldier-4', './temp/More_vis/Soldier-3', './temp/More_vis/Springhill-2', './temp/More_vis/Springhill-5', './temp/More_vis/Springhill-4', './temp/More_vis/Springhill-3', './temp/More_vis/Springhill-1', './temp/More_vis/Cooperstown-1', './temp/More_vis/Cooperstown-2', './temp/More_vis/Cooperstown-5', './temp/More_vis/Cooperstown-4', './temp/More_vis/Cooperstown-3', './temp/More_vis/Pettigrew-2', './temp/More_vis/Pettigrew-5', './temp/More_vis/Pettigrew-4', './temp/More_vis/Pettigrew-3', './temp/More_vis/Pettigrew-1', './temp/More_vis/Applewold-1', './temp/More_vis/Applewold-5', './temp/More_vis/Applewold-2', './temp/More_vis/Applewold-3', './temp/More_vis/Applewold-4', './temp/More_vis/Beach-3', './temp/More_vis/Beach-4', './temp/More_vis/Beach-5', './temp/More_vis/Beach-2', './temp/More_vis/Beach-1']

    # # train_scenes = ['./temp_run3_succ_6-5-23-seed--3/More_vis/Angiola','./temp_run3_succ_6-5-23-seed--3/More_vis/Albertville','./temp_run3_succ_6-5-23-seed--3/More_vis/Ballou', './temp_run3_succ_6-5-23-seed--3/More_vis/Elmira']
    # # test_scenes = ['./temp_run3_succ_6-5-23-seed--3/More_vis/Angiola','./temp_run3_succ_6-5-23-seed--3/More_vis/Beach']
    # print("Train scenes are:",train_scenes)
    # print("Test scenes are",test_scenes)

    # labels = create_labels(train_scenes)
    # dataset = AVDDataset(
    #         labels,
    #     )

    #     # Split dataset into train and val
    # train_idx, val_idx = train_test_split(
    #         np.arange(len(dataset)), 
    #         test_size=0.2, 
    #         random_state=42,
    #         shuffle=True,
    #         stratify=dataset.targets
    #     )

    # val_dataset = Subset(dataset, val_idx)

    # val_samples_per_class = val_dataset.class_samples
    

    # val_criterion = Loss(loss_type="focal_loss", samples_per_class = val_samples_per_class, class_balanced=True)
    test_scenes = ['./temp/More_vis/Sumas-1', './temp/More_vis/Sumas-2', './temp/More_vis/Sumas-3', './temp/More_vis/Sumas-4', './temp/More_vis/Sumas-5', './temp/More_vis/Roxboro-1', './temp/More_vis/Roxboro-2', './temp/More_vis/Roxboro-3', './temp/More_vis/Roxboro-4', './temp/More_vis/Roxboro-5', './temp/More_vis/Eudora-1', './temp/More_vis/Eudora-2', './temp/More_vis/Eudora-3', './temp/More_vis/Eudora-4', './temp/More_vis/Eudora-5', './temp/More_vis/Arkansaw-1', './temp/More_vis/Arkansaw-2', './temp/More_vis/Arkansaw-3', './temp/More_vis/Arkansaw-4', './temp/More_vis/Arkansaw-5', './temp/More_vis/Convoy-1', './temp/More_vis/Convoy-2', './temp/More_vis/Convoy-3', './temp/More_vis/Convoy-4', './temp/More_vis/Convoy-5', './temp/More_vis/Ribera-1', './temp/More_vis/Ribera-2', './temp/More_vis/Ribera-3', './temp/More_vis/Ribera-4', './temp/More_vis/Ribera-5', './temp/More_vis/Sanctuary-1', './temp/More_vis/Sanctuary-2', './temp/More_vis/Sanctuary-3', './temp/More_vis/Sanctuary-4', './temp/More_vis/Sanctuary-5', './temp/More_vis/Silas-1', './temp/More_vis/Silas-2', './temp/More_vis/Silas-3', './temp/More_vis/Silas-4', './temp/More_vis/Silas-5', './temp/More_vis/Bowlus-1', './temp/More_vis/Bowlus-2', './temp/More_vis/Bowlus-3', './temp/More_vis/Bowlus-4', './temp/More_vis/Bowlus-5', './temp/More_vis/Cooperstown-1', './temp/More_vis/Cooperstown-2', './temp/More_vis/Cooperstown-3', './temp/More_vis/Cooperstown-4', './temp/More_vis/Cooperstown-5', './temp/More_vis/Delton-1', './temp/More_vis/Delton-2', './temp/More_vis/Delton-3', './temp/More_vis/Delton-4', './temp/More_vis/Delton-5', './temp/More_vis/Rancocas-1', './temp/More_vis/Rancocas-2', './temp/More_vis/Rancocas-3', './temp/More_vis/Rancocas-4', './temp/More_vis/Rancocas-5', './temp/More_vis/Mesic-1', './temp/More_vis/Mesic-2', './temp/More_vis/Mesic-3', './temp/More_vis/Mesic-4', './temp/More_vis/Mesic-5', './temp/More_vis/Eagerville-1', './temp/More_vis/Eagerville-2', './temp/More_vis/Eagerville-3', './temp/More_vis/Eagerville-4', './temp/More_vis/Eagerville-5', './temp/More_vis/Goffs-1', './temp/More_vis/Goffs-2', './temp/More_vis/Goffs-3', './temp/More_vis/Goffs-4', './temp/More_vis/Goffs-5', './temp/More_vis/Bolton-1', './temp/More_vis/Bolton-2', './temp/More_vis/Bolton-3', './temp/More_vis/Bolton-4', './temp/More_vis/Bolton-5', './temp/More_vis/Mosinee-1', './temp/More_vis/Mosinee-2', './temp/More_vis/Mosinee-3', './temp/More_vis/Mosinee-4', './temp/More_vis/Mosinee-5', './temp/More_vis/Avonia-1', './temp/More_vis/Avonia-2', './temp/More_vis/Avonia-3', './temp/More_vis/Avonia-4', './temp/More_vis/Avonia-5', './temp/More_vis/Anaheim-1', './temp/More_vis/Anaheim-2', './temp/More_vis/Anaheim-3', './temp/More_vis/Anaheim-4', './temp/More_vis/Anaheim-5', './temp/More_vis/Azusa-1', './temp/More_vis/Azusa-2', './temp/More_vis/Azusa-3', './temp/More_vis/Azusa-4', './temp/More_vis/Azusa-5']

    print("Test scenes are",test_scenes)


    model = EmbedModel(vlad).cuda()
    model.load_state_dict(torch.load("best_model_vlad.pt"))
    model.eval()  # set the model to evaluation mode
    ##### Testing #####
    test_labels = create_labels(test_scenes)
    test_dataset = VisAVDDataset(
            test_labels,
            backbone_str='None',
        )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # _, y_pred_list, label_list = evaluate(model, test_loader)
    # y_pred_np_list = [tensor.cpu().numpy() for tensor in y_pred_list]
    # label_np_list = [tensor.cpu().numpy() for tensor in label_list]
    test_auc, test_iou = calculateAUC(model, test_loader)
    print("Test iou is",test_iou)
    print("The auc is:",test_auc)
