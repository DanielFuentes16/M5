import cv2
import glob
import os
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.logger import setup_logger
from detectron2.evaluation import COCOEvaluator
import time
setup_logger()
from KITTIMOTSLoader import get_KITTIMOTS_dicts

print('########################################################')
print('########################################################')
print('################## Load KittiMots ######################')
print('########################################################')
print('########################################################')
for d in ["train", "val"]:
    DatasetCatalog.register("kittimots-" + d, lambda d=d: get_KITTIMOTS_dicts(d))
    MetadataCatalog.get("kittimots-" + d).set(thing_classes=['Car', 'Pedestrian', 'DontCare'])

# Create config
print('########################################################')
print('########################################################')
print('################## Configuration #######################')
print('########################################################')
print('########################################################')
cfg = get_cfg()
cfg.merge_from_file("../../Week2/detectron2_repo/configs/COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
cfg.DATASETS.TRAIN = ("kittimots-train",)
cfg.DATASETS.TEST = ("kittimots-val",)
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = "detectron2://COCO-Detection/faster_rcnn_R_101_FPN_3x/137851257/model_final_f6e8b1.pkl"
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)

#Evaluation with COCO
print('########################################################')
print('########################################################')
print('#################### Evaluation ########################')
print('########################################################')
print('########################################################')
evaluator = COCOEvaluator("kittimots-train", cfg, False, output_dir="./output/")
trainer.test(cfg, trainer.model, evaluators=[evaluator])



