import os
import sys
import glob
import cv2
import torch
from tqdm import tqdm
from detectron2.evaluation import COCOEvaluator
import numpy as np

from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.utils.logger import setup_logger
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer

setup_logger()

file_dir = os.path.dirname(__file__)
print(file_dir)
sys.path.append(file_dir)

from KITTIMOTSLoader import get_KITTIMOTS_dicts
import MaskConfiguration as mk

inference = True
class MaskCNN_MOTS(object):
    def run(self, argv):

        #conf = "R50-C4"
        if len(argv) == 1:
            #configuration = mk.MaskConfiguration().get_Configuration(conf)
            exit(0)
        else:
            conf = argv[1]
            configuration = mk.MaskConfiguration().get_Configuration(conf)
        PATH_RESULTS = './Results-{}/'.format(conf)
        PATH_TRAIN = '/home/mcv/datasets/KITTI-MOTS/training/image_02/'
        PATH_TEST = '/home/mcv/datasets/KITTI/data_object_image_2/testing/image_2'
        os.makedirs(PATH_RESULTS, exist_ok=True)
        print("////////////////////////////////////////////////////////")
        print("////////////////////Configuration///////////////////////")
        print("////////////////////////////////////////////////////////")
        print(configuration[0])
        print(configuration[1])
        print("////////////////////////////////////////////////////////")
        print("////////////////////////////////////////////////////////")
        print("////////////////////////////////////////////////////////")
        dataset = get_KITTIMOTS_dicts("train")
        dataset_val = get_KITTIMOTS_dicts("val")
        for d in ["train", "val"]:
            DatasetCatalog.register("fcnn-mots" + d, lambda d=d: get_KITTIMOTS_dicts(d))
            MetadataCatalog.get("fcnn-mots" + d).set(thing_classes=['Car', 'DontCare', 'Pedestrian'])
        # Inference
        cfg = get_cfg()

        cfg.merge_from_file(configuration[0])
        metadata =  MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
        cfg.DATASETS.TRAIN = ('fcnn-motstrain',)
        cfg.DATASETS.TEST = ('fcnn-motsval',)
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model

        cfg.OUTPUT_DIR = PATH_RESULTS
        cfg.MODEL.WEIGHTS = configuration[1]

        #Train parameters
        cfg.DATALOADER.NUM_WORKERS = 2
        cfg.SOLVER.IMS_PER_BATCH = 4
        cfg.SOLVER.BASE_LR = 0.00025
        itersToFullDataset = len(dataset) // cfg.SOLVER.IMS_PER_BATCH + 1
        itersMultiplier = 1
        itersDivider = 1
        cfg.SOLVER.MAX_ITER = itersMultiplier * itersToFullDataset // itersDivider
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3

        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
        trainer = DefaultTrainer(cfg)
        trainer.resume_or_load(resume=False)
        trainer.train()

        # Evaluation
        evaluator = COCOEvaluator('fcnn-motsval', cfg, False, output_dir='./output{}'.format(conf))
        trainer = DefaultTrainer(cfg)
        trainer.test(cfg, model, evaluators=[evaluator])

if __name__ == '__main__':
    MaskCNN_MOTS().run(sys.argv)


