import csv
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
import itertools
import subprocess
import numpy as np
import argparse
import shutil

# parser = argparse.ArgumentParser(description='Description of your program.')
# parser.add_argument('--input_dataset', type=str, default='my_pairs.txt', help='Path to the input pairs file.')
# args = parser.parse_args()
# input_dataset = args.input_dataset

def visual_iou(thresholds, ious_list, auc):
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True)
    plt.subplots_adjust(left=0.3)
    plt.title('IOU vs Threshold')
    plt.xlabel('Thresholds')
    plt.ylabel('IOU')
    plt.plot(thresholds, ious_list, '-o', color='red', linewidth=2, label='IOU')
    plt.fill_between(thresholds, ious_list, color='blue', alpha=0.1)
    plt.text(0.5, 0.5, 'AUC = %0.2f' % auc)
    plot_name = "_superglue.png"
    file_path = os.path.join('vis', plot_name)  
    plt.savefig(file_path)
    plt.close()

scenes_dir = "temp/More_vis"
# specific_scene = ['temp/More_vis/Goffs']
specific_scene = ['temp/More_vis/Nemacolin-5',
 'temp/More_vis/Eastville-1',
 'temp/More_vis/Stokes-1',
 'temp/More_vis/Ribera-3',
 'temp/More_vis/Applewold-3',
 'temp/More_vis/Hometown-2',
 'temp/More_vis/Eudora-5',
 'temp/More_vis/Sanctuary-4',
 'temp/More_vis/Dunmor-2',
 'temp/More_vis/Pettigrew-4',
 'temp/More_vis/Spencerville-4',
 'temp/More_vis/Hainesburg-2',
 'temp/More_vis/Kerrtown-3',
 'temp/More_vis/Oyens-5',
 'temp/More_vis/Monson-4',
 'temp/More_vis/Roeville-4',
 'temp/More_vis/Spotswood-2',
 'temp/More_vis/Micanopy-3',
 'temp/More_vis/Angiola-2',
 'temp/More_vis/Nimmons-5',
 'temp/More_vis/Silas-4',
 'temp/More_vis/Anaheim-2',
 'temp/More_vis/Mifflintown-2',
 'temp/More_vis/Sumas-4',
 'temp/More_vis/Oyens-1',
 'temp/More_vis/Spencerville-1',
 'temp/More_vis/Pettigrew-5',
 'temp/More_vis/Convoy-4',
 'temp/More_vis/Eagerville-2',
 'temp/More_vis/Placida-1',
 'temp/More_vis/Capistrano-5',
 'temp/More_vis/Hometown-4',
 'temp/More_vis/Superior-1',
 'temp/More_vis/Mobridge-3',
 'temp/More_vis/Avonia-4',
 'temp/More_vis/Mesic-3',
 'temp/More_vis/Stanleyville-1',
 'temp/More_vis/Delton-5',
 'temp/More_vis/Silas-5',
 'temp/More_vis/Mosinee-1',
 'temp/More_vis/Nuevo-4',
 'temp/More_vis/Nimmons-1',
 'temp/More_vis/Beach-3',
 'temp/More_vis/Hominy-4',
 'temp/More_vis/Ribera-4',
 'temp/More_vis/Micanopy-2',
 'temp/More_vis/Spotswood-4',
 'temp/More_vis/Rosser-4',
 'temp/More_vis/Andover-4',
 'temp/More_vis/Delton-4',
 'temp/More_vis/Albertville-4',
 'temp/More_vis/Eagerville-4',
 'temp/More_vis/Hambleton-4',
 'temp/More_vis/Monson-5',
 'temp/More_vis/Dryville-1',
 'temp/More_vis/Pleasant-4',
 'temp/More_vis/Crandon-5',
 'temp/More_vis/Annawan-1',
 'temp/More_vis/Parole-5',
 'temp/More_vis/Hominy-3',
 'temp/More_vis/Nimmons-2',
 'temp/More_vis/Andover-3',
 'temp/More_vis/Ballou-2',
 'temp/More_vis/Rosser-3',
 'temp/More_vis/Arkansaw-3',
 'temp/More_vis/Cantwell-5',
 'temp/More_vis/Hillsdale-2',
 'temp/More_vis/Parole-1',
 'temp/More_vis/Bolton-4',
 'temp/More_vis/Denmark-2',
 'temp/More_vis/Eagerville-5',
 'temp/More_vis/Azusa-2',
 'temp/More_vis/Ballou-4',
 'temp/More_vis/Elmira-5',
 'temp/More_vis/Swormville-1',
 'temp/More_vis/Ballou-1',
 'temp/More_vis/Shelbiana-4',
 'temp/More_vis/Haxtun-5',
 'temp/More_vis/Mosquito-5',
 'temp/More_vis/Placida-4',
 'temp/More_vis/Silas-1',
 'temp/More_vis/Elmira-2',
 'temp/More_vis/Sands-4',
 'temp/More_vis/Sodaville-3',
 'temp/More_vis/Sands-2',
 'temp/More_vis/Shelbiana-3',
 'temp/More_vis/Hainesburg-5']
thresholds = [i / 100 for i in range(0, 101, 5)]
ious_list = []

max_num_matches = 0
gt_labels = {}

# for scene_dir in glob.glob(os.path.join(scenes_dir, '*')):
for scene_dir in specific_scene:
    floor_dirs = glob.glob(os.path.join(scene_dir, '[0-9]*'))  # Match directories with numeric names
    for floor_dir in floor_dirs:
        groundtruth_file = os.path.join(floor_dir, 'saved_obs', 'GroundTruth.csv')
        with open(groundtruth_file, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # Skip the header
            for row in reader:
                pair = tuple(row[:2])
                label = int(row[2])
                gt_labels[pair] = label

max_num_matches = 0

for scene_dir in specific_scene:
    floor_dirs = glob.glob(os.path.join(scene_dir, '[0-9]*'))  # Match directories with numeric names
    for floor_dir in floor_dirs:
        saved_obs_dir = os.path.join(floor_dir, 'saved_obs')
        pair_dir = os.path.join(floor_dir, 'my_pairs.txt')
        npz_dir = os.path.join(floor_dir, 'npz_result')

        # Write the pairs to a text file
        with open(pair_dir, 'w') as f:
            with open(os.path.join(saved_obs_dir, 'GroundTruth.csv'), 'r') as file2:
                reader = csv.reader(file2)
                next(reader)  # Skip the header
                for row in reader:
                    file_1 = os.path.basename(row[0])
                    file_2 = os.path.basename(row[1])
                    f.write('{} {}\n'.format(file_1, file_2))

        # Run the matching algorithm
        subprocess.run(["python", "match_pairs.py", "--input_pairs", pair_dir, "--input_dir", saved_obs_dir, "--output_dir", npz_dir])

        # Load predicted number of matches for each image pair from the generated npz files
        npz_files = glob.glob(os.path.join(npz_dir, '*.npz'))
        for npz_file in npz_files:
            npz = np.load(npz_file)
            pair = tuple([f.split('.')[0] for f in npz_file.split('/')[-1].split('_')[:2]])
            num_matches = np.sum(npz['matches'] > -1)
            max_num_matches = max(max_num_matches, num_matches)

for threshold_percent in thresholds:
    threshold = int(max_num_matches * threshold_percent)
    predicted_labels = {}
    tp = fp = fn = 0
    # Load predicted number of matches again
    
    # for scene_dir in glob.glob(os.path.join(scenes_dir, '*')):
    for scene_dir in specific_scene:
        floor_dirs = glob.glob(os.path.join(scene_dir, '[0-9]*'))  # Match directories with numeric names
        for floor_dir in floor_dirs:
            npz_dir = os.path.join(floor_dir, 'npz_result')
            npz_files = glob.glob(os.path.join(npz_dir, '*.npz'))
            for npz_file in npz_files:
                npz = np.load(npz_file)
                base_dir = os.path.dirname(npz_file)
                base_dir = os.path.dirname(base_dir)  # Go up one directory to '0' from 'npz_result'
                saved_obs_dir = os.path.join(base_dir, 'saved_obs')

                base_filename = os.path.basename(npz_file).split('_matches')[0]  # Remove the '_matches.npz' part

                parts = base_filename.split('_')  # Split the remaining string on '_'
                file_1 = '_'.join(parts[:3]) + '.png'  # Join the first three parts and append '.png'
                file_2 = '_'.join(parts[3:]) + '.png'  # Join the last three parts and append '.png'

                file_1_path = os.path.join("./",saved_obs_dir, file_1)
                file_2_path = os.path.join("./",saved_obs_dir, file_2)

                pair = (file_1_path, file_2_path)
                num_matches = np.sum(npz['matches'] > -1)
                if num_matches > threshold:
                    predicted_labels[pair] = 1
                else:
                    predicted_labels[pair] = 0

            # Calculate TP, FN, FP for all scenes and add them together
            for pair, predicted_label in predicted_labels.items():
                if pair in gt_labels:
                    gt_label = gt_labels[pair]
                    if predicted_label == 1 and gt_label == 1:
                        tp += 1
                    elif predicted_label == 1 and gt_label == 0:
                        fp += 1
                    elif predicted_label == 0 and gt_label == 1:
                        fn += 1
    print("tp:",tp)
    print("\nfp:",fp)
    print("\nfn:",fn)
    # Calculate IOU for the threshold
    intersection = tp
    union = tp + fp + fn
    iou = intersection / union if union != 0 else 0
    ious_list.append(iou)

# Calculate AUC
auc = np.trapz(ious_list, thresholds)

# Visualize IOU
visual_iou(thresholds, ious_list, auc)
print("iou list:",ious_list)