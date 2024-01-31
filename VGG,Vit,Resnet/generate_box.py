import shutil
import cv2
import os
import json
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.logger import setup_logger
from tqdm import tqdm
import glob

def get_boxes(target_path, img_path, model_path):
    if not os.path.exists(target_path):
        os.mkdir(target_path)
    else:
        for filename in os.listdir(target_path):
            file_path = os.path.join(target_path, filename)
            try:
                  if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                  elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
            except Exception as e:
                  print('Failed to delete %s. Reason: %s' % (file_path, e))
    
    box_conf_threshold = 0.2
    device_num = 1

    setup_logger()
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(
        'COCO-Detection/faster_rcnn_R_50_DC5_1x.yaml'))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = box_conf_threshold
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.01
    if device_num < 0:
        cfg.MODEL.DEVICE = 'cpu'
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1235
    cfg.MODEL.WEIGHTS = model_path
    cfg.INPUT.MIN_SIZE_TEST = 300

    model = DefaultPredictor(cfg)

    # loop through all images in img_path
    #for img in tqdm(os.listdir(img_path)):
    for img in tqdm(os.listdir(img_path)):
        # Only get images that start with "best_color_"
        if not img.startswith('best_color_'):
            continue
        # Get images
        image = cv2.imread(img_path +'/'+ img)
        image = cv2.resize(image, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
        # Get boxes
        corners = create_boxes(image, model)

        # Store boxes as JSON
        with open(target_path+'/'+img[:-4]+'.json', 'w') as f:
            json.dump(corners, f)
    
    return True

def create_boxes(image, predictor):
    boxes = None

    outputs = predictor(image)
    boxes = outputs["instances"].pred_boxes

    corners = {}
    count = 0

    for i in range(len(boxes)):
        box = boxes[i].tensor[0].cpu().numpy().astype(str)

        top = (box[0], box[1])
        bottom = (box[2], box[3])

        #box = Image.new('L', (IMAGE_SIZE,IMAGE_SIZE))
        #tmp = ImageDraw.Draw(box)
        #tmp.rectangle([top, bottom], fill='white')
        #trans = T.ToTensor()
        #pred_boxes[count] = trans(box)
        corners[count] = {
            'top': top,
            'bottom': bottom
        }
        count += 1
    
    return corners
scenes_dir = "temp_run3_succ_6-5-23-seed--3/More_vis"
model_path = "checkpoints/detector_lvis_thor.pth"
print("hi from outside")
for scene_dir in glob.glob(os.path.join(scenes_dir, '*')):
    print("hi from inside")
    floor_dirs = glob.glob(os.path.join(scene_dir, '[0-9]*'))  # Match directories with numeric names
    for floor_dir in floor_dirs:
        print("hi from floor inside")
        print("hi")
        saved_obs_dir = os.path.join(floor_dir, 'saved_obs')
        target_dir = os.path.join(saved_obs_dir,'boxes')
        get_boxes(target_dir,saved_obs_dir,model_path) 
