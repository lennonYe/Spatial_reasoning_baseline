from ablation import ablation
from createDataset import createDataset
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

    """# Create config directory
    create_dir('configs/', clean=True)
    
    configs = []
    print("\n\n==============Generating YAML files==============")
    # For each combination of hyperparameters, generate a YAML file
    for backbone in backbones:
        for attention in with_attention:
            for concat in concat_csr:
                for balanced in class_balanced:
                    print("Generating YAML file for %s, attention=%s, concat=%s, balanced=%s" % (backbone, attention, concat, balanced))
                    # Create a dictionary for the YAML file
                    config = {
                        'seed': SEED,
                        'lr': 1e-4,
                        'batch_size': 64,
                        'backbone_str': backbone,
                        'class_balanced': balanced,
                        'with_attention': attention,
                        'concat_csr': concat
                    }"""

    #                 configs.append(config)

    #                 # Create the YAML file
    #                 """with open('configs/%s_att%s_csr%s_bal%s.yaml' % (backbone, attention, concat, balanced), 'w') as file:
    #                     documents = yaml.dump(config, file)"""
    # gc.collect()


    """DEBUG"""
    configs = [{
        'lr': 1e-4,
        'batch_size': 64,
        'backbone_str': 'vgg',
        'class_balanced': True,
        'with_attention': False,
        'concat_csr': False
    }]
    print("\n\n==============Running ablation study==============")
    # For each configuration, run the ablation study on the deep learning methods
    start = time.time()
    ablation(configs, SEED, data_dir)
    end = time.time()
    print(f'Ablation took {(end - start) / 60:.0f} minutes and {(end - start) % 60:.0f} seconds')
    gc.collect()
    configs = [{
        'lr': 1e-4,
        'batch_size': 64,
        'backbone_str': 'vit',
        'class_balanced': True,
        'with_attention': False,
        'concat_csr': False
    }]
    ablation(configs, SEED, data_dir)
    gc.collect()
    configs = [{
        'lr': 1e-4,
        'batch_size': 64,
        'backbone_str': 'resnet',
        'class_balanced': True,
        'with_attention': False,
        'concat_csr': False
    }]
    ablation(configs, SEED, data_dir)
    gc.collect()
    print("\n\n==============Running Feature Matching==============")
    # Todo: implement SuperGlue and RANSAC (include vis; can use code from ablation.py for vis)
    #Generate auc plot for superglue for both train and test set
    # subprocess.run(["python", "superglue.py","--input_dataset", "test"])
    # subprocess.run(["python", "superglue.py","--input_dataset", "train"])

    # sift_ransac_instance = SIFT_RANSAC(path)
    # sift_ransac_instance.run()