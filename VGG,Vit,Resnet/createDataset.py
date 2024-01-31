from get_features import get_features
from get_boxes import get_boxes
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, help="dataset directory")

args = parser.parse_args()

model_path = 'checkpoints/detector_lvis_thor.pth'

def create_dir(dir, clean=False):
    try:
        if not os.path.exists(dir):
            os.makedirs(dir)
        elif clean:
            for file in os.listdir(dir):
                os.remove(dir + file)
    except Exception as e:
        raise e

def createDataset(path):
    # Create the boxes

    # Create target directory
    create_dir(os.path.join(path, 'boxes/'), clean=True)

    img_path = os.path.join(path, 'images/')
    target_path = os.path.join(path, 'boxes/')

    if not get_boxes(target_path, img_path, model_path):
        raise Exception("Error in get_boxes")

    # Create the features
    if not get_features(path):
        raise Exception("Error in get_features")
    
    print("Done")