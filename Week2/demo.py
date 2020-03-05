# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import cv2
import random

# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import torch, torchvision

# Check if version of torch is 1.4 the one compatible with detectron2
print(torch.__version__)

cfg = get_cfg()
cfg.merge_from_file("./detectron2_repo/configs/COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7 # set threshold for this model
cfg.MODEL.WEIGHTS = "detectron2://COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x/137849621/model_final_a6e10b.pkl"
from detectron2.modeling import build_model
model = build_model(cfg) # returns a torch.nn.Module
model.train(False)
img = cv2.imread("./input.jpg")
print(img)
img = np.transpose(img,(2,0,1))
img_tensor = torch.from_numpy(img)
inputs = [{"image":img_tensor}, {"image":img_tensor}]
outputs = model(inputs) #error may happen here
print(outputs)
