from inference_script import inference
# from createDataset import createDataset
import argparse
import os
import yaml
import gc
import time
import subprocess
from ransac import SIFT_RANSAC
from pytorch_lightning import seed_everything

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

if __name__ == "__main__":
    path = args.path
    
    backbones = ['csr', 'resnet', 'vgg', 'vit']
    with_attention = [True, False]
    concat_csr = [True, False]
    class_balanced = [True, False]

    data_dir = './temp/More_vis/'
    check_point_dir = './model_weights/Resnet.ckpt'

    configs = [{
        'lr': 1e-4,
        'batch_size': 64,
        'backbone_str': 'resnet',
        'class_balanced': True,
        'with_attention': False,
        'concat_csr': False
    }]
    print("\n\n==============Running ablation study==============")
    # For each configuration, run the ablation study on the deep learning methods
    start = time.time()
    inference(configs, SEED, data_dir,)
    end = time.time()
    print(f'Inference took {(end - start) / 60:.0f} minutes and {(end - start) % 60:.0f} seconds')