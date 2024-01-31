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
# from info_nce import InfoNCE
from info_nce_updated import InfoNCE
import csv
from collections import defaultdict
from sklearn.model_selection import train_test_split
import random
from PIL import Image
from scipy.spatial import KDTree
from sklearn.linear_model import LogisticRegression
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
def contrastive_loss(z1, image_name, positive_pairs, negative_pairs, temperature=0.75):
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

        positive_extra_img_names = random.sample(positive_pairs[query_name], 1)
        negative_extra_img_names = random.sample(negative_pairs[query_name], 1)

        positive_extra_images = [normalize(Image.open(img_name).convert("RGB")).unsqueeze(0) for img_name in positive_extra_img_names]
        negative_extra_images = [normalize(Image.open(img_name).convert("RGB")).unsqueeze(0) for img_name in negative_extra_img_names]

        positive_extra_images = torch.cat(positive_extra_images)
        negative_extra_images = torch.cat(negative_extra_images)
        # print("shape of positive extra", positive_extra_images.shape)
        # print("shape of negativeextra", negative_extra_images.shape)
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
    print("features_list_img1 is",features_list_img1)
    print("label_list:",labels_list)

    # Concatenate along the 0-axis to form a single array
    features_list_img1 = np.concatenate(features_list_img1, axis=0)
    features_list_img2 = np.concatenate(features_list_img2, axis=0)
    labels_list = np.concatenate(labels_list, axis=0)

    return features_list_img1, features_list_img2, labels_list

def find_extreme_embeddings(data):
    min_dist = float('inf')
    max_dist = float('inf')
    closest_pair = None

    for i in range(len(data)):
        for j in range(i+1, len(data)):
            dist = np.linalg.norm(np.array(data[i]) - np.array(data[j]))
            if dist < min_dist:
                min_dist = dist
                closest_pair = (data[i], data[j])
            if dist < max_dist:
                max_dist = dist

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
root_dir = "./RayCastDataset"
# testing_folder_path = "datasets/realworld/testing/images"
# train_label_folder = "datasets/realworld/training/training_csv.csv"
test_label_folder = "datasets/realworld/testing/testing_csv.csv"
positive_pairs,negative_pairs = load_pairs("MasterGroundTruth.csv")
batch_size = 64
# train_loader = get_dataloader(training_folder_path,train_label_folder, batch_size,True,"test")
test_loader = get_dataloader(root_dir, 1 ,positive_pairs,negative_pairs, "test")
train_loader = get_dataloader(root_dir,batch_size,positive_pairs,negative_pairs,"train")
logistic_train_loader = get_dataloader(root_dir,batch_size,positive_pairs,negative_pairs,"Logistic train")
epochs = 35
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
        loss = contrastive_loss(proj_z1,image_name,positive_pairs,negative_pairs)
        # loss = contrastive_loss(proj_z1, proj_z2,0.75)
        total_loss += loss.item()
        # Backpropagate and update the model parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    scheduler.step()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{epochs}], Avg. Loss: {avg_loss:.4f}")



# simclr_model.eval()  # Set the model to evaluation mode
# total_loss = 0.0
# total_batches = len(test_loader)
# thresholds = [i / 100 for i in range(0, 101, 10)]
# ious_list = []
# true_positives = 0
# false_positives = 0
# false_negatives = 0
# max_similarity = 1.0
# embeddings = []
# labels_list = []
# with torch.no_grad():
#     for batch_idx, batch in enumerate(test_loader):
#         img1, img2, label, _, _ = batch
#         proj_z1, proj_z2 = simclr_model(img1, img2)
#         embeddings.extend(proj_z1.cpu().numpy())
#         embeddings.extend(proj_z2.cpu().numpy())
#         labels_list.extend(label.cpu().numpy())
#         labels_list.extend(label.cpu().numpy())

# kd_tree = KDTree(embeddings)

# # for every embeddings search KDtree
# ious_list = []
# for threshold_percent in thresholds:
#     true_positives = 0
#     false_positives = 0
#     false_negatives = 0
#     threshold = float(max_similarity * threshold_percent)
#     print("Current threshold is:", threshold)

#     for idx, embedding in enumerate(embeddings):
#         #use kdtree find nearest neighbor
#         distances, indices = kd_tree.query(embedding.reshape(1, -1), k=2)  # k=2because nearest neighbor is itself
#         nearest_distance = distances[0][1]
#         nearest_idx = indices[0][1]

#         prediction = 1 if nearest_distance < threshold else 0
#         true_label = labels_list[idx]

#         if prediction == 1 and true_label == 1:
#             true_positives += 1
#         elif prediction == 1 and true_label == 0:
#             false_positives += 1
#         elif prediction == 0 and true_label == 1:
#             false_negatives += 1

#     iou = true_positives / (true_positives + false_positives + false_negatives)
#     ious_list.append(iou)

# auc = np.trapz(ious_list, thresholds)









simclr_model.eval()
features_img1, features_img2, labels = extract_features(logistic_train_loader, simclr_model)
combined_features = np.hstack((features_img1, features_img2))
clf = LogisticRegression(max_iter=10000)
# full dataset is loaded
clf.fit(combined_features, labels)

# class ClassificationModel(nn.Module):
#     def __init__(self, base_encoder):
#         super(ClassificationModel, self).__init__()
#         self.base_encoder = base_encoder
#         self.linear, self.classification_head = self.get_classification_head()
#         self.flatten = nn.Flatten()  # Add this line to define the flatten operation
#         self.base_encoder.eval()

#     def get_classification_head(self):
#         linear = []
#         fc = []
#         linear.append(nn.Linear(512,512))
#         fc.append(nn.ReLU(inplace=True))
#         fc.append(nn.BatchNorm1d(512))
#         fc.append(nn.Linear(512,256))
#         fc.append(nn.ReLU(inplace=True))
#         fc.append(nn.BatchNorm1d(256))
#         fc.append(nn.Linear(256,2))
#         fc.append(nn.Softmax(dim=1))
#         return nn.Sequential(*linear), nn.Sequential(*fc)
    
#     def forward(self,img1,img2):
#         input1 = self.base_encoder(img1)
#         input2 = self.base_encoder(img2)
#         if len(input1.shape) == 1:
#             input1 = input1.unsqueeze(0)
#             input2 = input2.unsqueeze(0)
#         input1 = self.flatten(input1)
#         input2 = self.flatten(input2)
#         out1 = self.linear(input1)
#         out2 = self.linear(input2)
#         joined = torch.pow(out1 - out2, 2)
#         out = self.classification_head(joined)
#         return out

# classification_model = ClassificationModel(simclr_model.base_encoder)


# classification_dataloader = logistic_train_loader # Change this to your actual classification dataloader

# # Setup criterion and optimizer for the classification training
# classification_criterion = nn.CrossEntropyLoss()
# classification_optimizer = torch.optim.Adam(classification_model.parameters(), lr=0.001)

# # Fine-tuning the Classifier
# classification_epochs = 35
# for epoch in range(classification_epochs):
#     classification_model.train()
#     total_loss = 0.0
#     for batch in classification_dataloader: 
#         # Forward pass
#         img1, img2, label, _, _ = batch
#         outputs = classification_model(img1,img2)
#         loss = classification_criterion(outputs, label)
#         # Backward and optimize
#         classification_optimizer.zero_grad()
#         loss.backward()
#         classification_optimizer.step()
#         total_loss += loss.item()
#     avg_loss = total_loss / len(classification_dataloader)
#     print(f"Classification Epoch [{epoch+1}/{classification_epochs}], Loss: {avg_loss:.4f}")

# classification_model.eval()
# total_loss = 0.0
# total_batches = len(test_loader)
# thresholds = [i / 100 for i in range(0, 101, 5)]
# ious_list = []
# true_positives = 0
# false_positives = 0
# false_negatives = 0
# max_threshold = 1.0
# for threshold_percent in thresholds:
#     true_positives = 0
#     false_positives = 0
#     false_negatives = 0
#     threshold = float(max_threshold * threshold_percent)
#     print("Current threshold is:",threshold)
#     with torch.no_grad():  # No need to calculate gradients during testing
#         for batch_idx, batch in enumerate(test_loader):
#             img1,img2,label,image_1,image_2= batch
#             predictions = (classification_model(img1, img2))[:,1].cpu().numpy()
#             print("First predictions",predictions)
#             # predictions = torch.nn.functional.softmax(predictions, dim=1)[:, 1].cpu().numpy()
#             # print("Second predictions",predictions)

#             # Concatenate the embeddings along the last dimension (features)
#             # print("Current image1 is:",image_1)
#             # print("Current image2 is:",image_2)
#             # print("Current label is:",label)
#             # print("logistic prediction is:",logistic_predictions)
#             # print("Number of tp", true_positives)
#             # print("number of fp", false_positives)
#             # print("number of fn", false_negatives)
#             binary_predictions = (predictions >= threshold).astype(np.float32)
#             binary_predictions = torch.tensor(binary_predictions)
#             true_positives += ((binary_predictions == 1) & (label == 1)).sum().item()
#             false_positives += ((binary_predictions == 1) & (label == 0)).sum().item()
#             false_negatives += ((binary_predictions == 0) & (label == 1)).sum().item()
#             # print("binary_predictions is:",binary_predictions)
#             # print("Number of tp", true_positives)
#             # print("number of fp", false_positives)
#             # print("number of fn", false_negatives)
#     iou = true_positives / (true_positives + false_positives + false_negatives)
#     ious_list.append(iou)

# auc = np.trapz(ious_list, thresholds)

# # Visualize IOU
# visual_iou(thresholds, ious_list, auc)
# avg_loss = total_loss / total_batches
# print(f"Average Loss on Test Set: {avg_loss:.4f}")
# iou = true_positives / (true_positives + false_positives + false_negatives)
# print(f"Intersection over Union (IOU): {iou:.4f}")



# Testing
total_loss = 0.0
total_batches = len(test_loader)
thresholds = [i / 100 for i in range(0, 101, 2)]
ious_list = []
true_positives = 0
false_positives = 0
false_negatives = 0
max_threshold = 1.0
for threshold_percent in thresholds:
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    threshold = float(max_threshold * threshold_percent)
    print("Current threshold is:",threshold)
    with torch.no_grad():  # No need to calculate gradients during testing
        for batch_idx, batch in enumerate(test_loader):
            img1,img2,label,image_1,image_2= batch
            embed_1 = simclr_model(img1)
            embed_2 = simclr_model(img2)
            embed_1_np = embed_1.cpu().numpy()
            embed_2_np = embed_2.cpu().numpy()
        
            # Concatenate the embeddings along the last dimension (features)
            combined_features = np.hstack((embed_1_np, embed_2_np))
            proba_predictions = clf.predict_proba(combined_features)
            logistic_predictions = proba_predictions[:, 1]
            # print("Current image1 is:",image_1)
            # print("Current image2 is:",image_2)
            # print("Current label is:",label)
            # print("logistic prediction is:",logistic_predictions)
            # print("Number of tp", true_positives)
            # print("number of fp", false_positives)
            # print("number of fn", false_negatives)
            binary_predictions = (logistic_predictions >= threshold).astype(np.float32)
            binary_predictions = torch.tensor(binary_predictions)
            true_positives += ((binary_predictions == 1) & (label == 1)).sum().item()
            false_positives += ((binary_predictions == 1) & (label == 0)).sum().item()
            false_negatives += ((binary_predictions == 0) & (label == 1)).sum().item()
            # print("binary_predictions is:",binary_predictions)
            # print("Number of tp", true_positives)
            # print("number of fp", false_positives)
            # print("number of fn", false_negatives)
    iou = true_positives / (true_positives + false_positives + false_negatives)
    ious_list.append(iou)

auc = np.trapz(ious_list, thresholds)

# Visualize IOU
visual_iou(thresholds, ious_list, auc)
avg_loss = total_loss / total_batches
print(f"Average Loss on Test Set: {avg_loss:.4f}")
iou = true_positives / (true_positives + false_positives + false_negatives)
print(f"Intersection over Union (IOU): {iou:.4f}")

# simclr_model.eval()  # Set the model to evaluation mode
# total_loss = 0.0
# total_batches = len(test_loader)
# thresholds = [i / 100 for i in range(0, 101, 10)]
# ious_list = []
# true_positives = 0
# false_positives = 0
# false_negatives = 0
# max_similarity = 1.0
# embeddings = []
# labels_list = []
# with torch.no_grad():
#     for batch_idx, batch in enumerate(test_loader):
#         img1, img2, label, _, _ = batch
#         proj_z1, proj_z2 = simclr_model(img1, img2)
#         embeddings.extend(proj_z1.cpu().numpy())
#         embeddings.extend(proj_z2.cpu().numpy())
#         labels_list.extend(label.cpu().numpy())
#         labels_list.extend(label.cpu().numpy())

# kd_tree = KDTree(embeddings)

# # for every embeddings search KDtree
# ious_list = []
# for threshold_percent in thresholds:
#     true_positives = 0
#     false_positives = 0
#     false_negatives = 0
#     threshold = float(max_similarity * threshold_percent)
#     print("Current threshold is:", threshold)

#     for idx, embedding in enumerate(embeddings):
#         #use kdtree find nearest neighbor
#         distances, indices = kd_tree.query(embedding.reshape(1, -1), k=2)  # k=2because nearest neighbor is itself
#         nearest_distance = distances[0][1]
#         nearest_idx = indices[0][1]

#         prediction = 1 if nearest_distance < threshold else 0
#         true_label = labels_list[idx]

#         if prediction == 1 and true_label == 1:
#             true_positives += 1
#         elif prediction == 1 and true_label == 0:
#             false_positives += 1
#         elif prediction == 0 and true_label == 1:
#             false_negatives += 1

#     iou = true_positives / (true_positives + false_positives + false_negatives)
#     ious_list.append(iou)

# auc = np.trapz(ious_list, thresholds)

# # Visualize IOU
# visual_iou(thresholds, ious_list, auc)
# avg_loss = total_loss / total_batches
# print(f"Average Loss on Test Set: {avg_loss:.4f}")
# iou = true_positives / (true_positives + false_positives + false_negatives)
# print(f"Intersection over Union (IOU): {iou:.4f}")

# Testing
# simclr_model.eval()  # Set the model to evaluation mode
# total_loss = 0.0
# total_batches = len(test_loader)
# thresholds = [i / 100 for i in range(0, 101, 10)]
# ious_list = []
# true_positives = 0
# false_positives = 0
# false_negatives = 0
# max_similarity = 1.0
# with torch.no_grad():  # No need to calculate gradients during testing
#     for batch_idx,batch in enumerate(test_loader):
#         img1,img2,label,image_1,image_2= batch
#         proj_z1, proj_z2 = simclr_model(img1, img2)

#         similarities = torch.matmul(F.normalize(proj_z1, dim=1), F.normalize(proj_z2, dim=1).t())
#         max_similarity = max(max_similarity, similarities.max().item())
# for threshold_percent in thresholds:
#     true_positives = 0
#     false_positives = 0
#     false_negatives = 0
#     threshold = float(max_similarity * threshold_percent)
#     print("Current threshold is:",threshold)
#     with torch.no_grad():  # No need to calculate gradients during testing
#         for batch_idx, batch in enumerate(test_loader):
#             img1,img2,label,image_1,image_2= batch
#             embed_1, embed_2 = simclr_model(img1, img2)
#             embed_1_normalized= F.normalize(embed_1, dim=1)
#             embed_2_normalized = F.normalize(embed_2, dim=1)
#             similarities = F.cosine_similarity(embed_1_normalized.unsqueeze(1), embed_2_normalized.unsqueeze(0), dim=2)
#             binary_predictions = (similarities >= threshold).float()
#             print("Current image1 is:",image_1)
#             print("Current image2 is:",image_2)
#             print("Current prediction is:",binary_predictions)
#             print("Current label is:",label)
#             print("similarity is:",similarities)
#             print("Number of tp", true_positives)
#             print("number of fp", false_positives)
#             print("number of fn", false_negatives)
#             true_positives += ((binary_predictions == 1) & (label == 1)).sum().item()
#             false_positives += ((binary_predictions == 1) & (label == 0)).sum().item()
#             false_negatives += ((binary_predictions == 0) & (label == 1)).sum().item()
#     iou = true_positives / (true_positives + false_positives + false_negatives)
#     ious_list.append(iou)

# auc = np.trapz(ious_list, thresholds)

# # Visualize IOU
# visual_iou(thresholds, ious_list, auc)
# avg_loss = total_loss / total_batches
# print(f"Average Loss on Test Set: {avg_loss:.4f}")
# iou = true_positives / (true_positives + false_positives + false_negatives)
# print(f"Intersection over Union (IOU): {iou:.4f}")


# Testing