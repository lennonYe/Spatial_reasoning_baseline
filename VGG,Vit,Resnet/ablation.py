import os
import re
import torch
import shutil
import numpy as np
import pandas as pd
from tqdm import tqdm
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
#from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins import DDPPlugin
from omegaconf import OmegaConf
from sklearn.model_selection import train_test_split
from pytorch_lightning import seed_everything
from NewLoader import AVDDataset, Subset
import matplotlib.pyplot as plt
from lightning import ClassificationModel
from visLoader import VisAVDDataset
import gc
torch.cuda.empty_cache()
gc.collect()

os.environ['MKL_THREADING_LAYER'] = 'GNU'
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

#parser = argparse.ArgumentParser()
#parser.add_argument('--conf', type=str, help="path config files")

#args = parser.parse_args()

def visualizeAUC(auc, iou, config_filename, backbone, balanced, attention,dataset_type):
    """
    Plot the IOU on a graph with the AUC
    """
    x = np.arange(0, 1.05, 0.05)

    # Set x and y axes limits
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True)
    plt.subplots_adjust(left=0.3)

    # Set plot title and axes labels
    plt.title('AUC')
    plt.xlabel('Thresholds')
    plt.ylabel('IOU')

    # Draw the plot
    plt.plot(x, iou, '-o', color='red', linewidth=2, label='IOU')

    # Fill in the area under the curve
    plt.fill_between(x, iou, color='blue', alpha=0.1)

    # Label the AUC
    plt.text(0.5, 0.5, 'AUC = %0.2f' % auc)

    # Write config
    plt.figtext(0, 0.55, 'Backbone: %s' % backbone, fontsize=8)
    plt.figtext(0, 0.5, 'Class Balanced: %s' % balanced, fontsize=8)
    plt.figtext(0, 0.45, 'With Attention: %s' % attention, fontsize=8)
    print("iou list is",iou)
    # Save the plot
    full_path = os.path.join('vis', dataset_type+'_'+config_filename)
    plt.savefig(full_path)

    plt.close()

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

def getPredictions(model, dataloader, csv_filename, class_balanced=False, concat_csr=False):
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
            # Get entry for csv
            # filename1 = filenames[0][0]
            # filename2 = filenames[1][0]

            # entry = [filename1, filename2, out.item(), label.item()]

            # csv_output.append(entry)
    
    # Save the csv
    # df = pd.DataFrame(csv_output)
    # df.to_csv('vis/'+csv_filename+'.csv', index=False, header=False)

    return pred, labels

def calculateAUC(model, dataloader, csv_filename, class_balanced=False, concat_csr=False):
    """
    Calculate the area under the curve using the IOUs calculated from each threshold from 0 to 1 with 0.05 increments
    """
    ious = []
    thresholds = np.arange(0, 1.05, 0.05)
    pred, labels = getPredictions(model, dataloader, csv_filename, class_balanced=class_balanced, concat_csr=concat_csr)

    for t in thresholds:
        iou = calculateIOU(pred, labels, threshold=t)
        ious.append(iou)
    
    print(f'IOUs: {ious}')
    auc = np.trapz(ious, thresholds)
    print(f'AUC: {auc}')
    return auc, ious

def create_labels(scene_names):
    labels = pd.DataFrame()
    for scene_dir in scene_names:
        for floor in os.listdir(scene_dir):
            if os.path.isdir(os.path.join(scene_dir, floor)) and re.search('[0-9]', floor):
                labels_path = os.path.join(scene_dir, floor, 'saved_obs', 'GroundTruth.csv')
                labels_df = pd.read_csv(labels_path)
                labels = pd.concat([labels, labels_df], ignore_index=True)
    return labels

def ablation(configs, seed, data_dir):
    SEED=seed
    seed_everything(SEED, workers = True)
    num_gpu = torch.cuda.device_count()
    print(f"Num Devices: {type(num_gpu)}, {num_gpu})")

    # Create vis directory
    if not os.path.exists('vis'):
        os.makedirs('vis')

    # Clean Log files
    if os.path.exists('lightning_logs'):
        for file in os.listdir('lightning_logs'):
            if file.startswith('version'):
                shutil.rmtree(os.path.join('lightning_logs', file))

    # Print total number of configs
    print(f'Total number of configs: {len(configs)}')
   
    scenes = [os.path.join(data_dir,scene_name) for scene_name in os.listdir(data_dir)]

    print(f'Total number of scenes: {len(scenes)}')
    print(f'scenes: {scenes}')

    # train_scenes, test_scenes = train_test_split(
    #     scenes,
    #     test_size=0.2,
    #     random_state=SEED,
    #     shuffle=True
    # )
    # train_scenes = ['./RayCastDataset/More_vis/Angiola','./RayCastDataset/More_vis/Albertville','./RayCastDataset/More_vis/Anaheim', './RayCastDataset/More_vis/Andover']
    # test_scenes = ['./RayCastDataset/More_vis/Adrian','./RayCastDataset/More_vis/Annawan']
    train_scenes = ['./temp/More_vis/Applewold','./temp/More_vis/Goffs','./temp/More_vis/Mesic', './temp/More_vis/Sanctuary','./temp/More_vis/Silas']
    test_scenes = ['./temp/More_vis/Anaheim']
    print("Train scenes are:",train_scenes)
    print("Test scenes are:",test_scenes)
    # Iterate through all the config files in the directory
    for conf in configs:
        print(f'Testing config: {conf}')
        
        # Make labels
        labels = create_labels(train_scenes)
        print("Total ones: ", sum([1 for i in labels["label"] if i == 1]))
        print("Total Zeros: ", sum([1 for i in labels["label"] if i == 0]))

        # Prepare datasets
        dataset = AVDDataset(
            labels=labels,
            backbone_str=conf["backbone_str"],
        )

        # Split dataset into train and val
        train_idx, val_idx = train_test_split(
            np.arange(len(dataset)), 
            test_size=0.2, 
            random_state=SEED,
            shuffle=True,
            stratify=dataset.targets
        )

        train_dataset = Subset(dataset, train_idx)
        val_dataset = Subset(dataset, val_idx)

        print("\nStarting training")
        model = ClassificationModel(train_dataset, val_dataset, conf["lr"], conf["batch_size"],
                conf["backbone_str"], conf["class_balanced"], conf["with_attention"], conf["concat_csr"], SEED)

        config_filename = 'DEBUG'

        # Create checkpoint callback directory for each config
        dir_name = f'./checkpoints/spatial-reasoning/{config_filename}'
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        
        # If checkpoints exist, delete them
        for file in os.listdir(dir_name):
            os.remove(os.path.join(dir_name, file)) 

        checkpoint_callback = ModelCheckpoint(
            dirpath=dir_name,
            # filename=f'{config_filename}' + '-{epoch:02d}-{val_loss:.4f}',
            filename=f'{config_filename}-{{epoch:02d}}',
            monitor='val_loss',
            mode='min',
            save_top_k=1,
        )

        trainer = Trainer(max_epochs=35, gpus=num_gpu, accelerator="ddp", callbacks=[checkpoint_callback], plugins=DDPPlugin(find_unused_parameters=True), amp_level='O2', precision=16)
        # trainer = Trainer(max_epochs=20, gpus=num_gpu, callbacks=[checkpoint_callback])
        print("\nFitting")
        trainer.fit(model)

        # Use the callback to get the best checkpoint
        best_model_path = checkpoint_callback.best_model_path
        print(f"best_model_path: {best_model_path}")

        # Load the best checkpoint
        best_model = ClassificationModel.load_from_checkpoint(best_model_path).to('cuda')

        test_labels = create_labels(test_scenes)

        # Prepare datasets
        test_dataset = VisAVDDataset(
            labels=test_labels,
            backbone_str=conf["backbone_str"],
        )

        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=8)

        # # Calculate AUC
        # print("\nCalculating AUC for train")
        # train_auc, train_iou = calculateAUC(best_model, train_loader, 'train_'+config_filename, conf["class_balanced"], conf["concat_csr"])
        if conf["backbone_str"] == "vit":
            config_filename = 'vit'
        elif conf["backbone_str"] == "resnet":
            config_filename = 'resnet'
        elif conf["backbone_str"] == "vgg":
            config_filename = "vgg"
        print("\nCalculating AUC for val")
        test_auc, test_iou = calculateAUC(best_model, test_loader, 'test_'+config_filename, conf["class_balanced"], conf["concat_csr"])

        # Visualize AUC
        if not os.path.exists('vis'):
            os.makedirs('vis')
        # visualizeAUC(train_auc, train_iou, config_filename, conf["backbone_str"], conf["class_balanced"], conf["with_attention"], 'train')
        visualizeAUC(test_auc, test_iou, config_filename, conf["backbone_str"], conf["class_balanced"], conf["with_attention"], 'test')

        # Clean up memory for next config
        del model
        del best_model
        del dataset
        del test_dataset
        # del train_loader
        del test_loader
        del train_dataset
        del val_dataset
        torch.cuda.empty_cache()
        gc.collect()
        # print(f'Finished testing config: {config}\n\n')
    
    print('Finished testing all configs!')
