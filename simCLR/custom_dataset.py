import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import glob as glob
import re
#dataset for running testing
class CustomImagePairDataset(Dataset):
    def __init__(self, root_folder,scenes):
        self.root_folder = root_folder
        self.scenes = scenes
        self.normalize = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize the image to (224, 224)
        transforms.ToTensor(),          # Convert the image to a PyTorch tensor
        transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # Mean values for normalization
        std=[0.229, 0.224, 0.225]    # Standard deviation values for normalization
        ),
        ])
        self.csv_dirs = []  # Create a list to store all csv directories
        for scene in self.scenes:
            scene_dir = os.path.join(self.root_folder, 'More_vis', scene)
            floor_dirs = [f for f in os.listdir(scene_dir) if os.path.isdir(os.path.join(scene_dir, f)) and re.match(r'^\d$', f)]
            for floor in floor_dirs:
                csv_dir = os.path.join(scene_dir, floor, 'saved_obs/GroundTruth.csv')
                self.csv_dirs.append(csv_dir)  # Append each csv directory to the list
        self.labels = pd.concat([pd.read_csv(csv_path) for csv_path in self.csv_dirs]).reset_index(drop=True)
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        # img1_path = os.path.join(self.root_folder, self.image_list[idx])
        # img2_path = os.path.join(self.root_folder, self.image_list[(idx + 1) % len(self.image_list)])
        image_1 = self.labels.iloc[idx]['image_1']
        image_2 = self.labels.iloc[idx]['image_2']
        img1_path = image_1
        img2_path = image_2
        img1 = Image.open(img1_path).convert('RGB')
        img2 = Image.open(img2_path).convert('RGB')
        label = self.labels.iloc[idx]['label']
        img1 = self.normalize(img1)
        img2 = self.normalize(img2)
        return img1, img2, label , image_1, image_2
    
#dataset for running training
class SimCLRDataset(Dataset):
    def __init__(self, root_dir,scenes,positive_pairs,negative_pairs):
        self.root_dir = root_dir
        self.scenes = scenes
        self.images = []
        self.normalize = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize the image to (224, 224)
        transforms.ToTensor(),          # Convert the image to a PyTorch tensor
        transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # Mean values for normalization
        std=[0.229, 0.224, 0.225]    # Standard deviation values for normalization
        ),
        ])
        self.augmented = transforms.Compose([
            # Randomly resize and crop the image
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomRotation(degrees=15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        for scene in self.scenes:
            scene_dir = os.path.join(self.root_dir, 'More_vis', scene)
            floor_dirs = [f for f in os.listdir(scene_dir) if os.path.isdir(os.path.join(scene_dir, f)) and re.match(r'^\d$', f)]
            for floor in floor_dirs:
                img_dir = os.path.join(scene_dir, floor, 'saved_obs')
                self.images += [os.path.join(img_dir, img) for img in os.listdir(img_dir) if re.match(r'best_color_\d+.png', img)]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        image = Image.open(image_path).convert("RGB")
        augmented_image = self.augmented(image)
        original_image = self.normalize(image)

        return augmented_image,original_image, image_path
    


