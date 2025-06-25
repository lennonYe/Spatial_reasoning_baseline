from ablation import ablation
# from createDataset import createDataset
import argparse
import os
import yaml
import gc
import time
import subprocess
# from ransac import SIFT_RANSAC
from pytorch_lightning import seed_everything
import yaml

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--skip', action='store_true', help="skip feature generation")
parser.add_argument('-p', '--path', type=str, help="dataset directory")
parser.add_argument('--seed', type=int, default=0, help='random seed (default: 0)')


args = parser.parse_args()
SEED = args.seed

seed_everything(SEED)
print(f'Using random seed {SEED}')

def create_dir(dir, clean=False):
    try:
        if not os.path.exists(dir):
            os.makedirs(dir)
        elif clean:
            for file in os.listdir(dir):
                os.remove(dir + file)
    except Exception as e:
        raise e

def load_config(config_path='config_vgg_vit_resnet.yaml'):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

if __name__ == "__main__":
    path = args.path
    config = load_config()

    # if not args.skip:
        # print("\n\n==============Generating boxes and features==============")
        # # For each Floor for every Scene in the dataset directory, generate the boxes and features
        # for dataset in os.listdir(path):
        #     if os.path.isdir(os.path.join(path, dataset)):
        #         # Scene
        #         # E.g., "Adrian"
                
        #         # For each floor...
        #         for floor in os.listdir(os.path.join(path, dataset)):
        #             if os.path.isdir(os.path.join(path, dataset, floor)):
        #                 # Floor
        #                 # E.g., "0"
        #                 print("Generating features for %s" % floor)
        #                 createDataset(os.path.join(path, dataset, floor, 'saved_obs'))
        # gc.collect()

    # Generate configuration YAML files
    
    backbones = ['csr', 'resnet', 'vgg', 'vit']
    with_attention = [True, False]
    concat_csr = [True, False]
    class_balanced = [True, False]

    data_dir = './temp/More_vis/'

    gc.collect()
    training_configs = [{
        'lr': config['learning_rate'],
        'batch_size': config['batch_size'],
        'backbone_str': config['backbone_str'],
        'class_balanced': config['class_balanced'],
        'with_attention': config['with_attention'],
        'concat_csr': config['concat_csr']
    }]
    ablation(training_configs, SEED, data_dir)
    gc.collect()