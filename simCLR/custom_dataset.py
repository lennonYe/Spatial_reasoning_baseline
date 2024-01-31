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
        # plt.figure()
        # plt.imshow(augmented_image.permute(1, 2, 0))
        # plt.title("Augmented Image")
        # plt.axis('off')
        # plt.show()

        # # Display original image
        # plt.figure()
        # plt.imshow(original_image.permute(1, 2, 0))
        # plt.title("Original Image")
        # plt.axis('off')
        # plt.show()
        return augmented_image,original_image, image_path
    

def get_dataloader(folder_path ,batch_size,positive_pair,negative_pair, type):
    if type == "train":
        dataset = SimCLRDataset(folder_path,['Sanctuary-5', 'Annawan-4', 'Albertville-5', 'Springhill-5', 'Soldier-5', 'Edgemere-1', 'Mifflintown-4', 'Spencerville-3', 'Greigsville-5', 'Bowlus-1', 'Maryhill-5', 'Monson-2', 'Eagerville-1', 'Woonsocket-1', 'Arkansaw-5', 'Colebrook-1', 'Quantico-2', 'Sumas-5', 'Kerrtown-1', 'Bowlus-3', 'Convoy-5', 'Greigsville-1', 'Woonsocket-2', 'Rancocas-2', 'Mosquito-1', 'Greigsville-3', 'Albertville-1', 'Scioto-4', 'Ballou-3', 'Soldier-4', 'Mosquito-4', 'Roane-5', 'Pablo-1', 'Swormville-2', 'Mosinee-5', 'Haxtun-3', 'Denmark-4', 'Goffs-3', 'Pleasant-5', 'Springhill-3', 'Rosser-2', 'Nemacolin-1', 'Cooperstown-5', 'Placida-3', 'Hainesburg-3', 'Applewold-1', 'Spencerville-5', 'Nuevo-3', 'Bowlus-5', 'Delton-3', 'Mosquito-3', 'Sasakwa-3', 'Bowlus-2', 'Avonia-1', 'Stokes-2', 'Mosinee-2', 'Anaheim-4', 'Nimmons-4', 'Quantico-5', 'Ballou-5', 'Mosinee-3', 'Swormville-4', 'Hometown-1', 'Brevort-2', 'Superior-4', 'Micanopy-5', 'Sands-1', 'Eastville-4', 'Azusa-5', 'Angiola-1', 'Andover-2', 'Roxboro-2', 'Roeville-5', 'Rancocas-5', 'Dryville-4', 'Denmark-5', 'Bolton-2', 'Dunmor-1', 'Shelbiana-1', 'Annawan-3', 'Annawan-2', 'Mobridge-5', 'Rosser-5', 'Rancocas-4', 'Sisters-1', 'Seward-1', 'Micanopy-1', 'Eastville-3', 'Capistrano-4', 'Parole-3', 'Roeville-2', 'Roane-3', 'Dryville-5', 'Stokes-5', 'Stokes-3', 'Sawpit-5', 'Sanctuary-1', 'Elmira-1', 'Azusa-3', 'Pleasant-2', 'Sasakwa-2', 'Stilwell-4', 'Convoy-2', 'Spotswood-1', 'Reyno-5', 'Sodaville-2', 'Colebrook-3', 'Parole-4', 'Pablo-4', 'Scioto-5', 'Nicut-1', 'Applewold-4', 'Delton-1', 'Mobridge-4', 'Bolton-5', 'Applewold-5', 'Edgemere-4', 'Maryhill-1', 'Nicut-5', 'Micanopy-4', 'Sumas-2', 'Angiola-3', 'Arkansaw-1', 'Dunmor-4', 'Nemacolin-4', 'Hambleton-3', 'Mosinee-4', 'Roxboro-1', 'Goffs-5', 'Hainesburg-1', 'Cooperstown-2', 'Spotswood-3', 'Hillsdale-4', 'Delton-2', 'Sawpit-3', 'Soldier-1', 'Sisters-3', 'Cooperstown-4', 'Oyens-4', 'Colebrook-5', 'Quantico-1', 'Hillsdale-5', 'Colebrook-2', 'Haxtun-1', 'Brevort-5', 'Andover-5', 'Maryhill-4', 'Brevort-4', 'Kerrtown-5', 'Adrian-4', 'Mifflintown-1', 'Dryville-3', 'Quantico-3', 'Superior-2', 'Pablo-2', 'Ribera-5', 'Nicut-4', 'Kerrtown-2', 'Hambleton-1', 'Eastville-2', 'Mesic-2', 'Pettigrew-3', 'Sasakwa-5', 'Nemacolin-3', 'Silas-3', 'Applewold-2', 'Hainesburg-4', 'Avonia-5', 'Mesic-1', 'Anaheim-1', 'Roxboro-4', 'Eagerville-3', 'Nuevo-5', 'Dunmor-5', 'Roeville-3', 'Goffs-4', 'Seward-3', 'Cantwell-1', 'Reyno-1', 'Sodaville-4', 'Goffs-2', 'Stanleyville-2', 'Mosquito-2', 'Bowlus-4', 'Edgemere-5', 'Beach-2', 'Crandon-4', 'Capistrano-2', 'Crandon-3', 'Cantwell-3', 'Stanleyville-4', 'Placida-2', 'Stokes-4', 'Adrian-3', 'Swormville-3', 'Sawpit-1', 'Albertville-3', 'Monson-3', 'Haxtun-4', 'Parole-2', 'Rancocas-1', 'Azusa-4', 'Quantico-4', 'Stilwell-2', 'Scioto-2', 'Woonsocket-4', 'Seward-5', 'Maryhill-2', 'Sodaville-1', 'Bolton-3', 'Angiola-4', 'Pettigrew-1', 'Azusa-1', 'Maryhill-3', 'Seward-4', 'Edgemere-2', 'Denmark-3', 'Adrian-2', 'Spencerville-2', 'Mesic-5', 'Sumas-1', 'Oyens-3', 'Hambleton-5', 'Sisters-2', 'Andover-1', 'Crandon-2', 'Anaheim-5', 'Crandon-1', 'Cantwell-4', 'Roane-2', 'Reyno-4', 'Hillsdale-1', 'Hillsdale-3', 'Cantwell-2', 'Pleasant-3', 'Dryville-2', 'Adrian-5', 'Mobridge-1', 'Superior-3', 'Rancocas-3', 'Pettigrew-2', '.DS_Store', 'Superior-5', 'Sanctuary-2', 'Pablo-3', 'Oyens-2', 'Mobridge-2', 'Hominy-1', 'Ribera-2', 'Sanctuary-3', 'Pablo-5', 'Stanleyville-3', 'Sawpit-4', 'Mifflintown-3', 'Springhill-1', 'Eudora-1', 'Bolton-1', 'Seward-2', 'Swormville-5', 'Hambleton-2', 'Beach-1', 'Convoy-3', 'Annawan-5', 'Nemacolin-2', 'Avonia-2', 'Rosser-1', 'Sasakwa-4', 'Nicut-2', 'Pleasant-1', 'Sasakwa-1', 'Adrian-1', 'Stilwell-5', 'Stilwell-3', 'Reyno-2', 'Greigsville-4', 'Beach-5', 'Woonsocket-3', 'Angiola-5', 'Goffs-1', 'Sumas-3', 'Springhill-2', 'Kerrtown-4', 'Albertville-2', 'Roxboro-5', 'Arkansaw-4', 'Nuevo-1', 'Edgemere-3', 'Sands-3', 'Elmira-3', 'Reyno-3', 'Greigsville-2', 'Roane-4', 'Scioto-1', 'Mesic-4', 'Eastville-5', 'Sodaville-5', 'Brevort-3', 'Cooperstown-3', 'Eudora-4', 'Roxboro-3', 'Stilwell-1', 'Nuevo-2', 'Ribera-1', 'Shelbiana-2', 'Roeville-1', 'Shelbiana-5', 'Brevort-1', 'Mifflintown-5', 'Stanleyville-5', 'Cooperstown-1', 'Sisters-5', 'Scioto-3', 'Haxtun-2', 'Soldier-2', 'Monson-1', 'Roane-1', 'Woonsocket-5', 'Silas-2', 'Eudora-2', 'Spotswood-5', 'Colebrook-4', 'Soldier-3', 'Hometown-3', 'Elmira-4', 'Denmark-1', 'Nicut-3', 'Sisters-4', 'Eudora-3', 'Capistrano-3', 'Nimmons-3', 'Hominy-2', 'Sawpit-2', 'Beach-4', 'Arkansaw-2', 'Convoy-1', 'Dunmor-3', 'Avonia-3', 'Sands-5', 'Anaheim-3', 'Placida-5', 'Hominy-5', 'Hometown-5', 'Capistrano-1', 'Springhill-4'],positive_pair,negative_pair)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    elif type == "test":
        dataset = CustomImagePairDataset(folder_path,['Nemacolin-5', 'Eastville-1', 'Stokes-1', 'Ribera-3', 'Applewold-3', 'Hometown-2', 'Eudora-5', 'Sanctuary-4', 'Dunmor-2', 'Pettigrew-4', 'Spencerville-4', 'Hainesburg-2', 'Kerrtown-3', 'Oyens-5', 'Monson-4', 'Roeville-4', 'Spotswood-2', 'Micanopy-3', 'Angiola-2', 'Nimmons-5', 'Silas-4', 'Anaheim-2', 'Mifflintown-2', 'Sumas-4', 'Oyens-1', 'Spencerville-1', 'Pettigrew-5', 'Convoy-4', 'Eagerville-2', 'Placida-1', 'Capistrano-5', 'Hometown-4', 'Superior-1', 'Mobridge-3', 'Avonia-4', 'Mesic-3', 'Stanleyville-1', 'Delton-5', 'Silas-5', 'Mosinee-1', 'Nuevo-4', 'Nimmons-1', 'Beach-3', 'Hominy-4', 'Ribera-4', 'Micanopy-2', 'Spotswood-4', 'Rosser-4', 'Andover-4', 'Delton-4', 'Albertville-4', 'Eagerville-4', 'Hambleton-4', 'Monson-5', 'Dryville-1', 'Pleasant-4', 'Crandon-5', 'Annawan-1', 'Parole-5', 'Hominy-3', 'Nimmons-2', 'Andover-3', 'Ballou-2', 'Rosser-3', 'Arkansaw-3', 'Cantwell-5', 'Hillsdale-2', 'Parole-1', 'Bolton-4', 'Denmark-2', 'Eagerville-5', 'Azusa-2', 'Ballou-4', 'Elmira-5', 'Swormville-1', 'Ballou-1', 'Shelbiana-4', 'Haxtun-5', 'Mosquito-5', 'Placida-4', 'Silas-1', 'Elmira-2', 'Sands-4', 'Sodaville-3', 'Sands-2', 'Shelbiana-3', 'Hainesburg-5'])
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    elif type == "Logistic train":
        dataset = CustomImagePairDataset(folder_path,["Adrian","Angiola","Andover","Anaheim"])
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


