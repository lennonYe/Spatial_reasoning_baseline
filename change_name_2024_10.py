import os

# The path to the directory containing the folders you want to rename


def change_folder_name():
    for i in range(1,6):
        directory_path = f"/Users/yimengye/Desktop/study/ai4ce_project/Spatial_reasoning_baseline/Dataset_2024_11/CoVISIONReasoningDataset_V1_EndR=0_pivotR=2_thresh=0.001_Seed={i}/temp/More_vis"
        # Iterate through each item in the directory
        for folder_name in os.listdir(directory_path):
            # Construct the full path to the item
            old_folder_path = os.path.join(directory_path, folder_name)

            # Check if the item is a directory
            if os.path.isdir(old_folder_path):
                # Create the new folder name by appending "-i"
                new_folder_path = os.path.join(directory_path, folder_name + f'-{i}')
                # Rename the folder
                os.rename(old_folder_path, new_folder_path)

def change_individual_csv_entries():
    for i in range(1,6):
        directory_path = f"/Users/yimengye/Desktop/study/ai4ce_project/Spatial_reasoning_baseline/Dataset_2024_11/CoVISIONReasoningDataset_V1_EndR=0_pivotR=2_thresh=0.001_Seed={i}/temp/More_vis"
        for folder_name in os.listdir(directory_path):
            # Iterate through each item in the directory
            if folder_name == '.DS_Store':
                continue
            scene_path = directory_path+'/'+folder_name
            numeric_floor = [path for path in os.listdir(scene_path) if path.isdigit()]
            for floor in numeric_floor:
                floor_path = scene_path + '/' + floor
                gt_path = floor_path + '/' + 'saved_obs' + '/' + 'GroundTruth.csv'
                updated_lines = []
                with open(gt_path,'r') as f1:
                    header = f1.readline()  # Read header
                    updated_lines.append(header.strip())
                    for line in f1:
                        line = line.strip().split(',')
                        part_one = line[0].split('/')
                        part_two = line[1].split('/')
                        part_one[-4] = part_one[-4] + f'-{i}'
                        part_two[-4] = part_two[-4] + f'-{i}'
                        line[0] = '/'.join(part_one)
                        line[1] = '/'.join(part_two)
                        updated_lines.append(','.join(line))
                        # import pdb;pdb.set_trace()
                with open(gt_path, 'w') as f1:
                    for updated_line in updated_lines:
                        f1.write(updated_line + '\n')

def integrate_master_csv():
    updated_lines = []
    write_path = './MasterGroundTruth.csv'
    for i in range(1,6):
        directory_path = f"/Users/yimengye/Desktop/study/ai4ce_project/Spatial_reasoning_baseline/Dataset_2024_11/CoVISIONReasoningDataset_V1_EndR=0_pivotR=2_thresh=0.001_Seed={i}"
        gt_path = directory_path + '/' + 'MasterGroundTruth.csv'
        with open(gt_path,'r') as f1:
            for line in f1:
                line = line.strip().split(',')
                part_one = line[0].split('/')
                part_two = line[1].split('/')
                part_one[-4] = part_one[-4] + f'-{i}'
                part_two[-4] = part_two[-4] + f'-{i}'
                line[0] = '/'.join(part_one)
                line[1] = '/'.join(part_two)
                updated_lines.append(','.join(line))
                # import pdb;pdb.set_trace()
        with open(write_path, 'w') as f1:
            for updated_line in updated_lines:
                f1.write(updated_line + '\n')


integrate_master_csv()


    

    