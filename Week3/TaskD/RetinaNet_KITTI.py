import os
import glob
import cv2
from tqdm import tqdm
from detectron2.evaluation import COCOEvaluator

from detectron2.structures import BoxMode
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.logger import setup_logger

from KITTIMOTSLoader import get_KITTIMOTS_dicts

for d in ["train", "val"]:
    DatasetCatalog.register("retina-kittimots" + d, lambda d=d: get_KITTIMOTS_dicts(d))
    MetadataCatalog.get("retina-kittimots" + d).set(thing_classes=['Car', 'Pedestrian', 'DontCare'])
kitti_metadata = MetadataCatalog.get("retina-kittimotstrain")

dataset_dicts = get_KITTIMOTS_dicts('train')

cfg = get_cfg()
cfg.merge_from_file("./detectron2_repo/configs/COCO-Detection/retinanet_R_101_FPN_3x.yaml")
cfg.DATASETS.TRAIN = ("retina-kittimotstrain",)
cfg.DATASETS.TEST = ()   # no metrics implemented for this dataset
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = "detectron2://COCO-Detection/retinanet_R_101_FPN_3x/138363263/model_final_59f53c.pkl"
cfg.SOLVER.IMS_PER_BATCH = 4
cfg.SOLVER.BASE_LR = 0.00025
itersToFullDataset = len(dataset_dicts) // cfg.SOLVER.IMS_PER_BATCH + 1
itersMultiplier = 5
itersDivider = 1
cfg.SOLVER.MAX_ITER = itersMultiplier * itersToFullDataset // itersDivider
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()

# load weights
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set the testing threshold for this model

# Set training data-set path
cfg.DATASETS.TEST = ("retina-kittimotsval", )


evaluator = COCOEvaluator("retina-kittimotsval", cfg, False, output_dir="./output/")
trainer.test(cfg, trainer.model, evaluators=[evaluator])
