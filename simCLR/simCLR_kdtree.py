import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.models as models
from torchvision.models.resnet import ResNet, resnet50,resnet18
from custom_dataset import get_dataloader
import matplotlib.pyplot as plt
import os
import numpy as np
import torchvision.transforms as T
from torch.utils.data import DataLoader
from custom_dataset import SimCLRDataset
# from info_nce import InfoNCE
from info_nce_updated import InfoNCE
import csv
from collections import defaultdict
from sklearn.model_selection import train_test_split
import random
from PIL import Image
from scipy.spatial import KDTree
from sklearn.linear_model import LogisticRegression
import glob
from torch.utils.data import random_split, DataLoader, SubsetRandomSampler

def visual_iou(thresholds, ious_list, auc):
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True)
    plt.subplots_adjust(left=0.3)
    plt.title('IOU vs Threshold')
    plt.xlabel('Threshold')
    plt.ylabel('IOU')
    plt.plot(thresholds, ious_list, '-o', color='red', linewidth=2, label='IOU')
    plt.fill_between(thresholds, ious_list, color='blue', alpha=0.1)
    plt.text(0.5, 0.5, 'AUC = %0.2f' % auc)
    plot_name = "simCLR.png"
    file_path = os.path.join('vis', plot_name)  
    if not os.path.exists('vis'):
        os.mkdir('vis')
    file_path = os.path.join('vis', plot_name)  
    plt.savefig(file_path)
    plt.close()

# def visual_loss(epochs,loss,type):

#     file_path = os.path.join('vis', plot_name)  
#     # Plot training and validation loss
#     if type == "train":
#         plt.figure(figsize=(10,6))
#         plt.plot(epochs, loss, '-o', label='Training Loss')
#         plt.title('Training and Validation Loss')
#         plt.xlabel('Epochs')
#         plt.ylabel('Loss')
#         plt.legend()
#         plt.grid(True)
#         plt.savefig(file_path)
#     elif type == "val":
#         plt.figure(figsize=(10,6))
#         plt.plot(epochs, loss, '-o', label='Validation Loss')
#         plt.title('Training and Validation Loss')
#         plt.xlabel('Epochs')
#         plt.ylabel('Loss')
#         plt.legend()
#         plt.grid(True)

def visual_loss(epochs,train_loss,val_loss):
    file_path = os.path.join('vis', "val_train_loss")  
    plt.figure(figsize=(10,6))

# Plot training and validation loss
    plt.plot(epochs, train_loss, '-o', label='Training Loss')
    plt.plot(epochs, val_loss, '-o', label='Validation Loss')
    # Labels and Title
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(file_path)



normalize =  T.Compose([
        T.Resize((224, 224)),  # Resize the image to (224, 224)
        T.ToTensor(),          # Convert the image to a PyTorch tensor
        T.Normalize(
        mean=[0.485, 0.456, 0.406],  # Mean values for normalization
        std=[0.229, 0.224, 0.225]    # Standard deviation values for normalization
        ),
        ])
def load_pairs(csv_path):
    positive_pairs = defaultdict(list)
    negative_pairs = defaultdict(list)

    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            img1, img2, label = row
            if int(label) == 1:
                positive_pairs[img1].append(img2)
                positive_pairs[img2].append(img1)
            else:
                negative_pairs[img1].append(img2)
                negative_pairs[img2].append(img1)
    
    return positive_pairs, negative_pairs

def find_max_neighbors(scenes_path,scene_name):
    max_count = 0
    print(scene_name)
    for scene in scene_name:
        scene_path = os.path.join(scenes_path, scene)
        if os.path.isdir(scene_path):  # Ensure it's a directory (scene)
            floor_dirs = glob.glob(os.path.join(scene_path, '[0-9]*'))  # Match dir
            for floor_dir in floor_dirs:
                    floor_path = os.path.join(floor_dir, 'saved_obs')
                    if os.path.exists(floor_path): 
                        images = [f for f in os.listdir(floor_path) if f.startswith("best_color") and f.endswith(".png")]
                        count = len(images)
                        if count > max_count:
                            max_count = count
    return max_count

def contrastive_loss(z1, image_name, positive_pairs, negative_pairs,filtered_positive,filtered_negative, temperature=0.75):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    total_loss = 0
    batch_size = z1.shape[0]
    valid_batch_size = batch_size
    loss = InfoNCE(temperature=temperature, reduction='mean', negative_mode='paired')
    for i in range(batch_size):
        query = z1[i].unsqueeze(0)
        query_name = image_name[i]
        # Find the positive and negative keys for the query within the batch
        positive_keys_list = [z1[j] for j in range(batch_size) if image_name[j] in positive_pairs[query_name]]
        negative_keys_list = [z1[j] for j in range(batch_size) if image_name[j] in negative_pairs[query_name]]
        if len(positive_pairs[query_name]) < 2 or len(negative_pairs[query_name]) < 2:
            # print("Skipping query image", query_name)
            # print("Positive_pairs for this image is:",positive_pairs[query_name])
            # print("Negative_pairs for this image is:",negative_pairs[query_name])
            valid_batch_size -= 1
            continue
        # positive_extra_img_names = random.sample(positive_pairs[query_name], 2)
        # negative_extra_img_names = random.sample(negative_pairs[query_name], 2)
        if len(filtered_positive[query_name]) == 0 or len(filtered_negative[query_name]) == 0:
            # Handle this case. Maybe continue to the next iteration or use some fallback logic
            valid_batch_size -= 1
            continue
        positive_extra_img_names = random.sample(filtered_positive[query_name], 1)
        negative_extra_img_names = random.sample(filtered_negative[query_name], 1)

        positive_extra_images = [normalize(Image.open(img_name).convert("RGB")).unsqueeze(0) for img_name in positive_extra_img_names]
        negative_extra_images = [normalize(Image.open(img_name).convert("RGB")).unsqueeze(0) for img_name in negative_extra_img_names]

        positive_extra_images = torch.cat(positive_extra_images)
        negative_extra_images = torch.cat(negative_extra_images)
        simclr_model.eval()
        positive_extra_embeds = simclr_model.base_encoder(positive_extra_images)  # Assuming you're using DataParallel
        positive_extra_embeds = positive_extra_embeds.view(positive_extra_embeds.size(0), -1)
        # print("shape of positive extra embed after baseencoder:",positive_extra_embeds.shape)
        positive_extra_embeds = simclr_model.projection_head(positive_extra_embeds)
        negative_extra_embeds = simclr_model.base_encoder(negative_extra_images)  # Assuming you're using DataParallel
        negative_extra_embeds = negative_extra_embeds.view(negative_extra_embeds.size(0), -1)
        # print("shape of negative extra embed after baseencoder:",negative_extra_embeds.shape)
        negative_extra_embeds = simclr_model.projection_head(negative_extra_embeds)
        simclr_model.train()

        if len(positive_keys_list) != 0:
            positive_keys = torch.stack(positive_keys_list).unsqueeze(0)
            # print("positive key 1 is",positive_keys.shape)
            # print("positive_extra_embeds after unsqueeze:",positive_extra_embeds.unsqueeze(0).shape)
            # positive_keys = torch.cat((positive_keys, positive_extra_embeds.unsqueeze(0)), dim=1)
        else:
            positive_keys = positive_extra_embeds.unsqueeze(0)
        if len(negative_keys_list) != 0:
            negative_keys = torch.stack(negative_keys_list).unsqueeze(0)
            # negative_keys = torch.cat((negative_keys, negative_extra_embeds.unsqueeze(0)), dim=1)
        else:
            negative_keys = negative_extra_embeds.unsqueeze(0)

        # Compute the InfoNCE loss for the query
        single_loss = loss(query, positive_keys, negative_keys)
        total_loss += single_loss
    print("total loss is:", total_loss)
    return total_loss / valid_batch_size if valid_batch_size > 0 else 0

def extract_features(dataloader, model):
    features_list_img1 = []
    features_list_img2 = []
    labels_list = []

    for batch in dataloader:
        img1, img2, label, _, _ = batch
        with torch.no_grad():
            features_img1= model(img1)
            features_img2 = model(img2)
        features_list_img1.append(features_img1.cpu().numpy())
        features_list_img2.append(features_img2.cpu().numpy())
        labels_list.append(label.cpu().numpy())

    # Concatenate along the 0-axis to form a single array
    features_list_img1 = np.concatenate(features_list_img1, axis=0)
    features_list_img2 = np.concatenate(features_list_img2, axis=0)
    labels_list = np.concatenate(labels_list, axis=0)

    return features_list_img1, features_list_img2, labels_list

def find_extreme_embeddings(data):
    min_dist = float('inf')
    max_dist = 0
    closest_pair = None

    for i in range(len(data)):
        for j in range(i+1, len(data)):
            dist = np.linalg.norm(np.array(data[i]) - np.array(data[j]))
            if dist < min_dist:
                min_dist = dist
                closest_pair = (data[i], data[j])
            if dist > max_dist:
                max_dist = dist
    print("min dist is",min_dist)
    print("Max dist is",max_dist)
    return min_dist, max_dist


class SimCLRModel(nn.Module):
    def __init__(self, base_encoder, projection_dim=128):
        super(SimCLRModel, self).__init__()
        self.base_encoder = base_encoder
        num_features = self.get_num_features()
        self.projection_head = nn.Sequential(
            nn.Linear(num_features, num_features),
            nn.BatchNorm1d(num_features),
            nn.ReLU(),
            nn.Linear(num_features, projection_dim),
            nn.BatchNorm1d(projection_dim)
        )
    def get_num_features(self):
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 224, 224)
            num_features = self.base_encoder(dummy_input).view(1, -1).shape[1]
        return num_features
    def forward(self, img1):
        z1 = self.base_encoder(img1)
        # Flatten the encoded features
        z1 = z1.view(z1.size(0), -1)
        # Compute the projections for contrastive loss
        if self.training:
            proj_z1 = self.projection_head(z1)
            return proj_z1
        else:
            return z1

# Load the base encoder
base_encoder = resnet18(pretrained=True) 
# Modify the base encoder to remove the final classification layer
base_encoder = nn.Sequential(*list(base_encoder.children())[:-1])
# Create the SimCLR model with the modified base encoder
simclr_model = SimCLRModel(base_encoder)
# data_dir = './temp/More_vis/'
# scenes = [os.path.join(data_dir,scene_name) for scene_name in os.listdir(data_dir)]
# train_scenes, test_scenes = train_test_split(
#         scenes,
#         test_size=0.2,
#         random_state=0,
#         shuffle=True
#     )
root_dir = "./temp"
positive_pairs,negative_pairs = load_pairs("MasterGroundTruth.csv")
batch_size = 64
dataset = SimCLRDataset(root_dir,['Sanctuary-5', 'Annawan-4', 'Albertville-5', 'Springhill-5', 'Soldier-5', 'Edgemere-1', 'Mifflintown-4', 'Spencerville-3', 'Greigsville-5', 'Bowlus-1', 'Maryhill-5', 'Monson-2', 'Eagerville-1', 'Woonsocket-1', 'Arkansaw-5', 'Colebrook-1', 'Quantico-2', 'Sumas-5', 'Kerrtown-1', 'Bowlus-3', 'Convoy-5', 'Greigsville-1', 'Woonsocket-2', 'Rancocas-2', 'Mosquito-1', 'Greigsville-3', 'Albertville-1', 'Scioto-4', 'Ballou-3', 'Soldier-4', 'Mosquito-4', 'Roane-5', 'Pablo-1', 'Swormville-2', 'Mosinee-5', 'Haxtun-3', 'Denmark-4', 'Goffs-3', 'Pleasant-5', 'Springhill-3', 'Rosser-2', 'Nemacolin-1', 'Cooperstown-5', 'Placida-3', 'Hainesburg-3', 'Applewold-1', 'Spencerville-5', 'Nuevo-3', 'Bowlus-5', 'Delton-3', 'Mosquito-3', 'Sasakwa-3', 'Bowlus-2', 'Avonia-1', 'Stokes-2', 'Mosinee-2', 'Anaheim-4', 'Nimmons-4', 'Quantico-5', 'Ballou-5', 'Mosinee-3', 'Swormville-4', 'Hometown-1', 'Brevort-2', 'Superior-4', 'Micanopy-5', 'Sands-1', 'Eastville-4', 'Azusa-5', 'Angiola-1', 'Andover-2', 'Roxboro-2', 'Roeville-5', 'Rancocas-5', 'Dryville-4', 'Denmark-5', 'Bolton-2', 'Dunmor-1', 'Shelbiana-1', 'Annawan-3', 'Annawan-2', 'Mobridge-5', 'Rosser-5', 'Rancocas-4', 'Sisters-1', 'Seward-1', 'Micanopy-1', 'Eastville-3', 'Capistrano-4', 'Parole-3', 'Roeville-2', 'Roane-3', 'Dryville-5', 'Stokes-5', 'Stokes-3', 'Sawpit-5', 'Sanctuary-1', 'Elmira-1', 'Azusa-3', 'Pleasant-2', 'Sasakwa-2', 'Stilwell-4', 'Convoy-2', 'Spotswood-1', 'Reyno-5', 'Sodaville-2', 'Colebrook-3', 'Parole-4', 'Pablo-4', 'Scioto-5', 'Nicut-1', 'Applewold-4', 'Delton-1', 'Mobridge-4', 'Bolton-5', 'Applewold-5', 'Edgemere-4', 'Maryhill-1', 'Nicut-5', 'Micanopy-4', 'Sumas-2', 'Angiola-3', 'Arkansaw-1', 'Dunmor-4', 'Nemacolin-4', 'Hambleton-3', 'Mosinee-4', 'Roxboro-1', 'Goffs-5', 'Hainesburg-1', 'Cooperstown-2', 'Spotswood-3', 'Hillsdale-4', 'Delton-2', 'Sawpit-3', 'Soldier-1', 'Sisters-3', 'Cooperstown-4', 'Oyens-4', 'Colebrook-5', 'Quantico-1', 'Hillsdale-5', 'Colebrook-2', 'Haxtun-1', 'Brevort-5', 'Andover-5', 'Maryhill-4', 'Brevort-4', 'Kerrtown-5', 'Adrian-4', 'Mifflintown-1', 'Dryville-3', 'Quantico-3', 'Superior-2', 'Pablo-2', 'Ribera-5', 'Nicut-4', 'Kerrtown-2', 'Hambleton-1', 'Eastville-2', 'Mesic-2', 'Pettigrew-3', 'Sasakwa-5', 'Nemacolin-3', 'Silas-3', 'Applewold-2', 'Hainesburg-4', 'Avonia-5', 'Mesic-1', 'Anaheim-1', 'Roxboro-4', 'Eagerville-3', 'Nuevo-5', 'Dunmor-5', 'Roeville-3', 'Goffs-4', 'Seward-3', 'Cantwell-1', 'Reyno-1', 'Sodaville-4', 'Goffs-2', 'Stanleyville-2', 'Mosquito-2', 'Bowlus-4', 'Edgemere-5', 'Beach-2', 'Crandon-4', 'Capistrano-2', 'Crandon-3', 'Cantwell-3', 'Stanleyville-4', 'Placida-2', 'Stokes-4', 'Adrian-3', 'Swormville-3', 'Sawpit-1', 'Albertville-3', 'Monson-3', 'Haxtun-4', 'Parole-2', 'Rancocas-1', 'Azusa-4', 'Quantico-4', 'Stilwell-2', 'Scioto-2', 'Woonsocket-4', 'Seward-5', 'Maryhill-2', 'Sodaville-1', 'Bolton-3', 'Angiola-4', 'Pettigrew-1', 'Azusa-1', 'Maryhill-3', 'Seward-4', 'Edgemere-2', 'Denmark-3', 'Adrian-2', 'Spencerville-2', 'Mesic-5', 'Sumas-1', 'Oyens-3', 'Hambleton-5', 'Sisters-2', 'Andover-1', 'Crandon-2', 'Anaheim-5', 'Crandon-1', 'Cantwell-4', 'Roane-2', 'Reyno-4', 'Hillsdale-1', 'Hillsdale-3', 'Cantwell-2', 'Pleasant-3', 'Dryville-2', 'Adrian-5', 'Mobridge-1', 'Superior-3', 'Rancocas-3', 'Pettigrew-2', 'Superior-5', 'Sanctuary-2', 'Pablo-3', 'Oyens-2', 'Mobridge-2', 'Hominy-1', 'Ribera-2', 'Sanctuary-3', 'Pablo-5', 'Stanleyville-3', 'Sawpit-4', 'Mifflintown-3', 'Springhill-1', 'Eudora-1', 'Bolton-1', 'Seward-2', 'Swormville-5', 'Hambleton-2', 'Beach-1', 'Convoy-3', 'Annawan-5', 'Nemacolin-2', 'Avonia-2', 'Rosser-1', 'Sasakwa-4', 'Nicut-2', 'Pleasant-1', 'Sasakwa-1', 'Adrian-1', 'Stilwell-5', 'Stilwell-3', 'Reyno-2', 'Greigsville-4', 'Beach-5', 'Woonsocket-3', 'Angiola-5', 'Goffs-1', 'Sumas-3', 'Springhill-2', 'Kerrtown-4', 'Albertville-2', 'Roxboro-5', 'Arkansaw-4', 'Nuevo-1', 'Edgemere-3', 'Sands-3', 'Elmira-3', 'Reyno-3', 'Greigsville-2', 'Roane-4', 'Scioto-1', 'Mesic-4', 'Eastville-5', 'Sodaville-5', 'Brevort-3', 'Cooperstown-3', 'Eudora-4', 'Roxboro-3', 'Stilwell-1', 'Nuevo-2', 'Ribera-1', 'Shelbiana-2', 'Roeville-1', 'Shelbiana-5', 'Brevort-1', 'Mifflintown-5', 'Stanleyville-5', 'Cooperstown-1', 'Sisters-5', 'Scioto-3', 'Haxtun-2', 'Soldier-2', 'Monson-1', 'Roane-1', 'Woonsocket-5', 'Silas-2', 'Eudora-2', 'Spotswood-5', 'Colebrook-4', 'Soldier-3', 'Hometown-3', 'Elmira-4', 'Denmark-1', 'Nicut-3', 'Sisters-4', 'Eudora-3', 'Capistrano-3', 'Nimmons-3', 'Hominy-2', 'Sawpit-2', 'Beach-4', 'Arkansaw-2', 'Convoy-1', 'Dunmor-3', 'Avonia-3', 'Sands-5', 'Anaheim-3', 'Placida-5', 'Hominy-5', 'Hometown-5', 'Capistrano-1', 'Springhill-4'],positive_pairs,negative_pairs)
# dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
# dataset_loader = get_dataloader(root_dir,batch_size,positive_pairs,negative_pairs,"train")
dataset_length = len(dataset)
train_length = int(0.8 * dataset_length)
val_length = dataset_length - train_length
train_dataset, val_dataset = random_split(dataset, [train_length, val_length])
train_image_names = [data[2] for data in train_dataset]  # Assuming the image name is the second element in your dataset
val_image_names = [data[2] for data in val_dataset]
train_positive_pairs = {img: [pair for pair in pairs if pair in train_image_names] 
                           for img, pairs in positive_pairs.items() if img in train_image_names}
train_negative_pairs = {img: [pair for pair in pairs if pair in train_image_names] 
                           for img, pairs in negative_pairs.items() if img in train_image_names}
val_positive_pairs = {img: [pair for pair in pairs if pair in val_image_names] 
                           for img, pairs in positive_pairs.items() if img in val_image_names}
val_negative_pairs = {img: [pair for pair in pairs if pair in val_image_names] 
                           for img, pairs in negative_pairs.items() if img in val_image_names}
train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size, shuffle=False)
test_loader = get_dataloader(root_dir, batch_size ,positive_pairs,negative_pairs, "test")
# train_loader = get_dataloader(root_dir,batch_size,positive_pairs,negative_pairs,"train")
epochs = 35
train_loss_list = []
val_loss_list = []
max_epoch = 0
best_train_loss = 4000
optimizer = torch.optim.Adam(simclr_model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader)*epochs)
# Training loop
for epoch in range(epochs):
    simclr_model.train()  # Set the model to training mode
    total_loss = 0.0
    for batch in train_loader:
        # print(len(batch))
        augmented_image,original_image,image_name = batch
        proj_z1= simclr_model(original_image)
        # Compute the contrastive loss between the projections
        loss = contrastive_loss(proj_z1,image_name,positive_pairs,negative_pairs,train_positive_pairs,train_negative_pairs)
        # loss = contrastive_loss(proj_z1, proj_z2,0.75)
        total_loss += loss.item()
        # Backpropagate and update the model parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    scheduler.step()

    avg_loss = total_loss / len(train_loader)
    train_loss_list.append(avg_loss)
    if epoch == 0 or avg_loss < best_train_loss:
        best_train_loss = avg_loss
        # Save the model weights
        max_epoch = epoch
        torch.save(simclr_model.state_dict(), "best_model.pth")
        print("Model saved for epoch",epoch)
    print(f"Epoch [{epoch+1}/{epochs}], Avg. Loss: {avg_loss:.4f}")

    # simclr_model.eval()  # Set the model to evaluation mode
    # with torch.no_grad():  # No gradient computation, speeds up the validation pass
    #     val_loss = 0.0
    #     for val_batch in val_loader:
    #         augmented_image, original_image, image_name = val_batch
    #         proj_z1 = simclr_model(original_image)
    #         print(proj_z1.shape)
    #         loss = contrastive_loss(proj_z1, image_name, positive_pairs, negative_pairs,val_positive_pairs,val_negative_pairs)
    #         val_loss += loss.item()
    # avg_val_loss = val_loss / len(val_loader)
    # val_loss_list.append(avg_val_loss)
    # print(f"Validation Loss: {avg_val_loss:.4f}")

    # Checkpointing the model

evaluation_model = SimCLRModel(base_encoder)

# Now, load the state dictionary
evaluation_model.load_state_dict(torch.load('best_model.pth'))
evaluation_model.eval()  # Set the model to evaluation mode
total_loss = 0.0
total_batches = len(test_loader)

ious_list = []
true_positives = 0
false_positives = 0
false_negatives = 0
embeddings = []
embeddings_set = []
embeddings_one = []
embeddings_two = []
img_names = []
labels_list = []
with torch.no_grad():
    for batch_idx, batch in enumerate(test_loader):
        img1, img2, label, img1_name, img2_name = batch
        embedding_one = evaluation_model(img1)
        embedding_two = evaluation_model(img2)
        for b in range(len(embedding_one)):
            if img1_name[b] not in img_names:
                embeddings.extend(embedding_one[b].cpu().numpy().reshape(1,-1))
                img_names.append(str(img1_name[b]))
            if img2_name[b] not in img_names:
                embeddings.extend(embedding_two[b].cpu().numpy().reshape(1,-1))
                img_names.append(str(img2_name[b]))
        embeddings_one.extend(embedding_one.cpu().numpy())
        embeddings_two.extend(embedding_two.cpu().numpy())
        labels_list.extend(label.cpu().numpy())
kd_tree = KDTree(embeddings)
# print("number of embeddings",len(embeddings))
embeddings_matrix = np.vstack(embeddings)
max_neighbors = len(embeddings) - 1

print("max neighbor is",max_neighbors)
# For ranking method
# thresholds = [i for i in range(1, max_neighbors + 1)]
ious_list = []
threshold_list = []
thresholds = [i/100 for i in range(1,101,5)]
thresholds.append(1)
centroid = np.mean(embeddings, axis=0)
# Step 2: Find the point farthest from the centroid
distances_to_centroid = np.linalg.norm(embeddings - centroid, axis=1)
index_to_centroid = np.argmax(distances_to_centroid)
farthest_point_from_centroid = embeddings[index_to_centroid]
# Step 3: Find the point farthest from the point found in step 2
distances_from_farthest_point = np.linalg.norm(embeddings - farthest_point_from_centroid, axis=1)
max_distance = np.max(distances_from_farthest_point)
print("First max distance:",max_distance)
min_distance,max_distance = find_extreme_embeddings(embeddings)
print("Second max_distance is: ",max_distance)
print("min distance is:",min_distance)

for threshold in thresholds:
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    # current_k = min_distance + threshold*(max_distance-min_distance)
    current_k = threshold*max_distance
    print("Current threshold is:", current_k)
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            img1, img2, label, img1_name, img2_name = batch
            embedding_one = evaluation_model(img1)
            embedding_two = evaluation_model(img2)
            indices = kd_tree.query_ball_point(embedding_one, current_k)
            # print(indices)
            # print(embedding_one)
            for b in range(len(embedding_one)):  # Iterate over each item in the batch
                # Convert the b-th embedding of embedding_two to a list and then get its index in the embeddings list
                nearest_embeddings = embeddings_matrix[indices[b]]
                if len(indices) == 1:
                    nearest_embeddings = nearest_embeddings.reshape(1,-1)
                img2_embedding = embedding_two[b].cpu().numpy()
                exist = np.any(np.all(nearest_embeddings == img2_embedding, axis=1))
                prediction = 1 if exist else 0 
                # Check if the embedding_two's index is among the KD-tree's returned indices for the b-th embedding_one
                # prediction = 1 if img2_embedding_idx in indices[b] else 0
                
                if prediction == 1 and label[b] == 1:
                    true_positives += 1
                elif prediction == 1 and label[b] == 0:
                    false_positives += 1
                elif prediction == 0 and label[b] == 1:
                    false_negatives += 1
    iou = true_positives / (true_positives + false_positives + false_negatives)
    threshold_list.append(current_k)
    ious_list.append(iou)
print("Max epoch is",max_epoch)
epoch_num = range(1, len(train_loss_list) + 1)
auc = np.trapz(ious_list, thresholds)
visual_iou(thresholds,ious_list,auc)
print("iou list is:",ious_list)
visual_loss(epoch_num,train_loss_list,val_loss_list)



# For ranking method
# for every embeddings search KDtree
# ious_list = []
# threshold_list = []
# print(len(embeddings))
# for threshold in thresholds:
#     true_positives = 0
#     false_positives = 0
#     false_negatives = 0
#     current_k = threshold
#     print("Current threshold is:", current_k)
#     with torch.no_grad():
#         for batch_idx, batch in enumerate(test_loader):
#             img1, img2, label, img1_name, img2_name = batch
#             embedding_one = simclr_model(img1)
#             embedding_two = simclr_model(img2)
#             distances, indices = kd_tree.query(embedding_one.cpu().numpy(), k=current_k)

#             for b in range(len(embedding_one)):  # Iterate over each item in the batch
#                 # Convert the b-th embedding of embedding_two to a list and then get its index in the embeddings list
#                 # img2_embedding_idx = embeddings.index(embedding_two[b].cpu().numpy().tolist())
#                 nearest_embeddings = embeddings_matrix[indices[b]]
#                 if current_k == 1:
#                     nearest_embeddings = nearest_embeddings.reshape(1,-1)
#                 img2_embedding = embedding_two[b].cpu().numpy()
#                 exist = np.any(np.all(nearest_embeddings == img2_embedding, axis=1))
#                 prediction = 1 if exist else 0 
#                 # Check if the embedding_two's index is among the KD-tree's returned indices for the b-th embedding_one
#                 # prediction = 1 if img2_embedding_idx in indices[b] else 0
                
#                 if prediction == 1 and label[b] == 1:
#                     true_positives += 1
#                 elif prediction == 1 and label[b] == 0:
#                     false_positives += 1
#                 elif prediction == 0 and label[b] == 1:
#                     false_negatives += 1
#     iou = true_positives / (true_positives + false_positives + false_negatives)
#     threshold_list.append(current_k)
#     ious_list.append(iou)
# threshes = np.array(thresholds)
# threshes = (threshes - 1)/(max_neighbors - 1)
# # thresh = (thresh - 1)/(max_neighbors - 1)
# print(threshes)
# auc = np.trapz(ious_list, threshes)
# visual_iou(threshes,ious_list,auc)
