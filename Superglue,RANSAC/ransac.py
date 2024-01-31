import argparse
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
import glob
import path
import scipy.io as sio
import os
import json
import imageio
import collections
import sys
from skimage.feature import plot_matches
from cv2 import DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
import pandas as pd
import matplotlib.ticker as ticker

class SIFT_RANSAC:

    def __init__(self, pathName):
        self.pathName = pathName
        self.imgPath = pathName
        self.groundTruthCSV = "/More_vis/*/*/saved_obs/groundtruth.csv"

    def sift_ransac_matches(self, pathName):
        images = []
        paths = []
        all_files = os.listdir(pathName)
        img_files = [file for file in all_files if file.startswith('best_color') and file.endswith('.png')]
        # list_dir = [int(file.split(".")[0]) for file in img_files]
        list_dir = [(file.split(".")[0]) for file in img_files]
        list_dir.sort()
        for i in range(len(list_dir)):
            path = pathName +'/' + str(list_dir[i]) + ".png"
            img = cv2.imread(path)
            paths.append(path)
            images.append(img)
        computed_inliers = {}
        ransac_dic = {}
        for i in range(len(paths)):
            ransac_dic[os.path.basename(paths[i])] = {}
            for j in range(len(paths)):
                if i != j:
                    new_path_j = "./" + paths[j]
                    new_path_i = "./" + paths[i]
                    pair_key = tuple(sorted([os.path.basename(paths[i]), os.path.basename(paths[j])]))
                    if pair_key in computed_inliers:
                        ransac_dic[os.path.basename(paths[i])][os.path.basename(paths[j])] = computed_inliers[pair_key]
                        continue
                    # sift_dic[i+1] = {}
                    img1 = images[i]
                    img2 = images[j]
                    img1 = cv2.resize(img1, (500,500))
                    img2 = cv2.resize(img2, (500,500))
                    gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
                    gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
                    sift = cv2.SIFT_create()

                        ##SIFT LOCALIZATION AND GETTING DESCRIPTOR THAT HAS HISTOGRAM ORIENTATION WITH SHAPE (#keyPoints*128):
                    kp1, des1 = sift.detectAndCompute(gray1, None)
                    kp2, des2 = sift.detectAndCompute(gray2, None)

                        ##Making brute force matcher that uses nearest neighbour
                    bf = cv2.BFMatcher()
                        ### Putting k = 1 for best one match 
                    matches = bf.knnMatch(des1,des2,k=2)  ##Matches gives : tuple of tuple matches
                    good = []
                    for m,n in matches:
                        if m.distance < 0.7*n.distance:
                            good.append(m)
                        ### Retrieving homography matrix using RANSAC and calculating inliers:
                    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
                    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
                    if len(src_pts)<4:
                        continue
                    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
                    matchesMask = mask.ravel().tolist()
                    inliers = sum(matchesMask) ## values with 1 in matched mask
                    if inliers>0:
                            computed_inliers[pair_key] = inliers
                            ransac_dic[os.path.basename(paths[i])][os.path.basename(paths[j])] = inliers
        # Flatten the nested dictionary to get all inlier values in a list
        all_inliers = [value for inner_dict in ransac_dic.values() for value in inner_dict.values()]

        # Find the maximum inliers
        max_inliers = max(all_inliers)
        min_inliers = min(all_inliers)
        return ransac_dic, images, max_inliers,min_inliers

    def calculate_auc(self, ious, threshes):
        auc = np.trapz(ious, threshes)
        return auc

    def visualize_ransac_result(self, threshes, ious, auc):
        stageRun = self.pathName.split("/")[-1]

        if stageRun == "testing":
            dataset_type = "test"
        elif stageRun == "training":
            dataset_type = "train"
        else:
            print("Saving SIFT baseline's result as vis/_sift_ransac.png")
            dataset_type = ""
         
        plt.figure()
        plt.grid(True)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.plot(threshes, ious, '-o', color = 'r', linewidth=2, label = "iou")
        plt.fill_between(threshes, ious, color='blue', alpha=0.1)
        plt.title("IOU vs Threshold")
        plt.text(0.5, 0.5, 'AUC = %0.2f' % auc)
        plt.xlabel("Thresholds")
        plt.ylabel("IOU")
        plt.text(0.05, 0.9, 'Baseline: RANSAC', transform=plt.gca().transAxes) 
        full_path = os.path.join('vis', dataset_type+'_sift_ransac.png')
        plt.savefig(full_path)
        plt.close()
        return 

    def test_ransac(self, img1, img2):
        
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
        ax[1].imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))

        plt.show()
        plt.close()
        
        img1 = cv2.resize(img1, (500,500))
        img2 = cv2.resize(img2, (500,500))
        gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
        sift = cv2.SIFT_create()
        
        ##SIFT LOCALIZATION AND GETTING DESCRIPTOR THAT HAS HISTOGRAM ORIENTATION WITH SHAPE (#keyPoints*128):
        kp1, des1 = sift.detectAndCompute(gray1, None)
        kp2, des2 = sift.detectAndCompute(gray2, None)

        ##Making brute force matcher that uses nearest neighbour
        bf = cv2.BFMatcher()

        ### Putting k = 2 so that we can apply ratio test as explained by D. Lowe 
        matches = bf.knnMatch(des1,des2,k=2)  ##Matches gives : tuple of tuple matches
        #matches = sorted(matches, key = lambda x:x[0].distance) ###indexing with x[0] because the tuple is like (x,).
        # If k = 2 then the tuple would be (x,y) i.e two matches for descriptor 1

        # Apply ratio test
        good = []
        good_idx = []
        i = 0
        for m,n in matches:
            if m.distance < 0.7*n.distance:
                good.append([m])
                good_idx.append(i)
            i = i + 1
        
        ### Retrieving homography matrix using RANSAC and calculating inliers:
        src_pts = np.float32([ kp1[m[0].queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m[0].trainIdx].pt for m in good ]).reshape(-1,1,2)
        
        if len(src_pts)<4:
            print("No matches found")
            return

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()

        inliers = [True if item == 1 else False for item in matchesMask]
        print("SIFT Matches:", len(matches))
        print("SIFT KD-2 Matches:", len(good))
        print("RANSAC Matches:", sum(inliers))
        if len(inliers)==0:
            print("No matches found")
            return
        outliers = [not x for x in inliers]
        inlier_idx = np.nonzero(inliers)[0]
        outlier_idx = np.nonzero(outliers)[0]
        
        src = np.array([m[0] for m in src_pts])
        dst = np.array([m[0] for m in dst_pts])
        src_correct = src[inlier_idx] 
        dst_correct = dst[inlier_idx]
        
        good_new = []
        for i in inlier_idx:
            good_new.append(good[i])
        
        siftMatches = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good_new,None,matchColor = (0, 0, 255), flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        plt.figure(figsize=(15, 15))
        plt.imshow(siftMatches)
        plt.show()
        plt.close()
        return 

    def run(self):
        pathName = self.imgPath
        scenes_path = os.path.join(self.imgPath, 'More_vis')
        # all_scenes = os.listdir(scenes_path)
        all_scenes = ['Nemacolin-5', 'Eastville-1', 'Stokes-1', 'Ribera-3', 'Applewold-3', 'Hometown-2', 'Eudora-5', 'Sanctuary-4', 'Dunmor-2', 'Pettigrew-4', 'Spencerville-4', 'Hainesburg-2', 'Kerrtown-3', 'Oyens-5', 'Monson-4', 'Roeville-4', 'Spotswood-2', 'Micanopy-3', 'Angiola-2', 'Nimmons-5', 'Silas-4', 'Anaheim-2', 'Mifflintown-2', 'Sumas-4', 'Oyens-1', 'Spencerville-1', 'Pettigrew-5', 'Convoy-4', 'Eagerville-2', 'Placida-1', 'Capistrano-5', 'Hometown-4', 'Superior-1', 'Mobridge-3', 'Avonia-4', 'Mesic-3', 'Stanleyville-1', 'Delton-5', 'Silas-5', 'Mosinee-1', 'Nuevo-4', 'Nimmons-1', 'Beach-3', 'Hominy-4', 'Ribera-4', 'Micanopy-2', 'Spotswood-4', 'Rosser-4', 'Andover-4', 'Delton-4', 'Albertville-4', 'Eagerville-4', 'Hambleton-4', 'Monson-5', 'Dryville-1', 'Pleasant-4', 'Crandon-5', 'Annawan-1', 'Parole-5', 'Hominy-3', 'Nimmons-2', 'Andover-3', 'Ballou-2', 'Rosser-3', 'Arkansaw-3', 'Cantwell-5', 'Hillsdale-2', 'Parole-1', 'Bolton-4', 'Denmark-2', 'Eagerville-5', 'Azusa-2', 'Ballou-4', 'Elmira-5', 'Swormville-1', 'Ballou-1', 'Shelbiana-4', 'Haxtun-5', 'Mosquito-5', 'Placida-4', 'Silas-1', 'Elmira-2', 'Sands-4', 'Sodaville-3', 'Sands-2', 'Shelbiana-3', 'Hainesburg-5']
         # Initialize the necessary variables for aggregation across scenes and floors
        iou_list = []
        iouMinThresh = 0
        ransac_thresh = {}
        minThresh = 5
        intersection = 0
        union = 0
        maxThresh = 0
        best_iou = 0
        scene_floor_ransac_dic = {}
        for scene in all_scenes:
            scene_path = os.path.join(scenes_path, scene)
            scene_floor_ransac_dic[scene] = {}
            if os.path.isdir(scene_path):  # Ensure it's a directory (scene)
                floor_dirs = glob.glob(os.path.join(scene_path, '[0-9]*'))  # Match dir
                for floor_dir in floor_dirs:
                    floor_path = os.path.join(floor_dir, 'saved_obs')
                    if os.path.exists(floor_path): 
                        ransac_dic, images,max_inliers,min_inliers = self.sift_ransac_matches(floor_path)
                        maxThresh = max(maxThresh,max_inliers)
                        minThresh = min(minThresh,min_inliers)
                        scene_floor_ransac_dic[scene][os.path.basename(floor_dir)] =  ransac_dic
        print(scene_floor_ransac_dic)
        max_matches = maxThresh  # Assuming maxThresh is the maximum number of matches

        # Generate threshold values evenly spaced between 0 and max_matches
        num_thresholds = 21  # You can change this number to control the number of thresholds
        threshes = np.linspace(0, max_matches, num_thresholds)
        print(threshes)
        for t in threshes:
            intersection = 0
            union = 0
            for scene in all_scenes:
                skip_count = 0
                scene_path = os.path.join(scenes_path, scene)
                if os.path.isdir(scene_path):  # Ensure it's a directory (scene)
                    floor_dirs = glob.glob(os.path.join(scene_path, '[0-9]*'))  # Match dir
                    for floor_dir in floor_dirs:
                            floor_path = os.path.join(floor_dir, 'saved_obs')
                            skip_count = 0
                            if os.path.exists(floor_path): 
                                gt_df = pd.read_csv(os.path.join(floor_path,"GroundTruth.csv"),skiprows = 1, names=["left","right", "value"])
                                for i in range(len(gt_df)):
                                    l = gt_df.iloc[i]["left"]
                                    r = gt_df.iloc[i]["right"]
                                    v = gt_df.iloc[i]["value"]
                                    if (scene in scene_floor_ransac_dic and os.path.basename(floor_dir) in scene_floor_ransac_dic[scene] and os.path.basename(l) in scene_floor_ransac_dic[scene][os.path.basename(floor_dir)] and os.path.basename(r) in scene_floor_ransac_dic[scene][os.path.basename(floor_dir)][os.path.basename(l)]):
                                        prediction =  scene_floor_ransac_dic[scene][os.path.basename(floor_dir)][os.path.basename(l)][os.path.basename(r)] > t
                                        label = v
                                        if prediction == True and label == True:
                                            intersection += 1
                                            union += 1
                                        elif prediction == False and label == True:
                                            union += 1
                                        elif prediction == True and label == False:
                                            union += 1
                                    else:
                                        skip_count += 1
                                print("INtersection now is:",intersection)
                                print("Union now is:",union)
                                print("For scene",scene,"floor:",os.path.basename(floor_dir),"We skip",skip_count)
            iou = intersection/union
            if iou > best_iou:
                best_iou = iou
                ransac_thresh_best = ransac_thresh
            iou_list.append((t,iou))

        threshes = [x[0] for x in iou_list]
        ious = [x[1] for x in iou_list]

        maxIdx = np.argmax(ious)
        max_iou = ious[maxIdx]
        thresh = threshes[maxIdx]

        threshes = np.array(threshes)
        threshes = (threshes - minThresh)/(maxThresh - minThresh)
        thresh = (thresh - minThresh)/(maxThresh - minThresh)

        auc = self.calculate_auc(ious, threshes)
        print(ious)
        self.visualize_ransac_result(threshes, ious, auc)


sift_ransac_instance = SIFT_RANSAC("temp")
sift_ransac_instance.run()