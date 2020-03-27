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
from MOTSChallengeLoader import get_MOTS_dicts
import MaskConfiguration as mk

inference = True
class MaskCNN_MOTS(object):
    def run(self, argv):
        useCityscapes = "CS" in sys.argv
        print("Has Cityscapes training data:" + str(useCityscapes))
        useMOTSChallenge = "MOTS" in sys.argv
        print("Has MOTSChallenge training data:" + str(useMOTSChallenge))
        useKITTIMOTS = "KITTI" in sys.argv
        print("Has KITTI-MOTS training data:" + str(useKITTIMOTS))
        conf = "R50-FPN" if useCityscapes is False else "R50-FPN-CS"
        print(conf)
        if useMOTSChallenge is False and useKITTIMOTS is False:
            exit(0)
        else:
            configuration = mk.MaskConfiguration().get_Configuration(conf)
        PATH_RESULTS = './Results-{}/'.format(conf)
        combo = ""
        if useCityscapes is True:
            combo += "-CS"
        if useKITTIMOTS is True:
            combo += "-KITTI"
        if useMOTSChallenge is True:
            combo += "-MOTS"
        PATH_RESULTS += combo
        os.makedirs(PATH_RESULTS, exist_ok=True)
        print("////////////////////////////////////////////////////////")
        print("////////////////////Configuration///////////////////////")
        print("////////////////////////////////////////////////////////")
        print(configuration[0])
        print(configuration[1])
        print("////////////////////////////////////////////////////////")
        print("////////////////////////////////////////////////////////")
        print("////////////////////////////////////////////////////////")
        
        trainDataCount = 0
        trainDatasetList = []

        if useKITTIMOTS is True:
            for d in ["train"]:
                DatasetCatalog.register("kitti" + d, lambda d=d: get_KITTIMOTS_dicts(d))
                MetadataCatalog.get("kitti" + d).set(thing_classes=['Car', 'Pedestrian'])
            kittiDataset = get_KITTIMOTS_dicts("train")
            trainDataCount += len(kittiDataset)
            trainDatasetList.append("kitti" + d)

        if useMOTSChallenge is True:
            for d in ["full"]:
                DatasetCatalog.register("mots" + d, lambda d=d: get_MOTS_dicts(d))
                MetadataCatalog.get("mots" + d).set(thing_classes=['Car', 'Pedestrian'])
            motsDataset = get_MOTS_dicts("full")
            trainDataCount += len(motsDataset)
            trainDatasetList.append("mots" + d)

        for d in ["val"]:
            DatasetCatalog.register("kitti" + d, lambda d=d: get_KITTIMOTS_dicts(d))
            MetadataCatalog.get("kitti" + d).set(thing_classes=['Car', 'Pedestrian'])
        # Inference
        cfg = get_cfg()

        cfg.merge_from_file(configuration[0])
        metadata = MetadataCatalog.get("kittival")
        cfg.DATASETS.TRAIN = tuple(trainDatasetList)
        cfg.DATASETS.TEST = ('kittival',)
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model

        cfg.OUTPUT_DIR = PATH_RESULTS
        cfg.MODEL.WEIGHTS = configuration[1]

        #Train parameters
        cfg.DATALOADER.NUM_WORKERS = 2
        cfg.SOLVER.IMS_PER_BATCH = 4
        cfg.SOLVER.BASE_LR = 0.00025
        itersToFullDataset = trainDataCount // cfg.SOLVER.IMS_PER_BATCH + 1
        itersMultiplier = 1
        itersDivider = 3
        cfg.SOLVER.MAX_ITER = itersMultiplier * itersToFullDataset // itersDivider
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2

        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
        trainer = DefaultTrainer(cfg)
        trainer.resume_or_load(resume=False)
        trainer.train()

        cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set the testing threshold for this model

        # Set training data-set path
        cfg.DATASETS.TEST = ('kittival',)

        # Evaluation
        evaluator = COCOEvaluator('kittival', cfg, False, output_dir='./output-{}'.format(conf) + combo)
        trainer.test(cfg, trainer.model, evaluators=[evaluator])

        print("Generating images with predictions...")

        dataset_val = get_KITTIMOTS_dicts("val")

        imagesToPredict = [97, 356, 527, 1293, 1875, 2121]

        cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
        predictor = DefaultPredictor(cfg)
        
        for img_idx in imagesToPredict:
            filePath = dataset_val[img_idx]['file_name']
            path, filename = os.path.split(filePath)
            # Make prediction
            im = cv2.imread(filePath)
            outputs = predictor(im)

            # Visualize the prediction in the image
            v = Visualizer(
                im[:, :, ::-1],
                metadata=metadata,
                scale=0.8,
                instance_mode=ColorMode.IMAGE)
            v = v.draw_instance_predictions(outputs["instances"].to("cpu"))

            os.makedirs(PATH_RESULTS, exist_ok=True)
            cv2.imwrite(PATH_RESULTS + filename, v.get_image()[:, :, ::-1])


if __name__ == '__main__':
    MaskCNN_MOTS().run(sys.argv)


