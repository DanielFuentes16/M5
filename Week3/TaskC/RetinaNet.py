import cv2
import glob
import os
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from KITTIMOTSLoader import get_KITTIMOTS_dicts
import torch
from tqdm import tqdm
from detectron2.evaluation import COCOEvaluator
import numpy as np

Debug = False

PATH_RESULTS = './Retinanet/'

# Create config
cfg = get_cfg()
cfg.merge_from_file("./detectron2_repo/configs/COCO-Detection/retinanet_R_101_FPN_3x.yaml")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
cfg.MODEL.WEIGHTS = "detectron2://COCO-Detection/retinanet_R_101_FPN_3x/138363263/model_final_59f53c.pkl"
#cfg.MODEL.DEVICE='cpu'

for d in ["train"]:
    DatasetCatalog.register("kittifull-retina", lambda d=d: get_KITTIMOTS_dicts(None, 1))
    MetadataCatalog.get("kittifull-retina").set(thing_classes=['Car', 'Pedestrian', 'DontCare'])

dataset_dicts = get_KITTIMOTS_dicts(None, 1)

# Inference
predictor = DefaultPredictor(cfg)
print("Generating Predictions")
predictions = []
for item in tqdm(dataset_dicts):
    # Make prediction
    im = cv2.imread(item['file_name'])
    outputs = predictor(im)
    cls_arr = outputs['instances']._fields['pred_classes'].cpu().numpy().tolist()
    scores_arr = outputs['instances']._fields['scores'].cpu().numpy().tolist()
    boxes_arr = outputs['instances']._fields['pred_boxes'].tensor.cpu().numpy().tolist()
    tmp_cls = []
    tmp_scores = []
    tmp_boxes = []
    for i, pack in enumerate(zip(cls_arr, scores_arr, boxes_arr)):
        cls_id = pack[0]
        score = pack[1]
        box = pack[2]
        if cls_id == 0:
            tmp_cls.append(1)
            tmp_scores.append(score)
            tmp_boxes.append(box)
        elif cls_id == 2:
            tmp_cls.append(0)
            tmp_scores.append(score)
            tmp_boxes.append(box)
    outputs['instances']._fields['pred_classes'] = torch.from_numpy(np.array(tmp_cls))
    outputs['instances']._fields['scores'] = torch.from_numpy(np.array(tmp_scores))
    outputs['instances']._fields['pred_boxes'].tensor = torch.from_numpy(np.array(tmp_boxes))
    predictions.append(outputs)

evaluator = COCOEvaluator("kittifull-retina", cfg, False, output_dir=PATH_RESULTS)
evaluator.reset()
evaluator.process(dataset_dicts, predictions)
evaluator.evaluate()