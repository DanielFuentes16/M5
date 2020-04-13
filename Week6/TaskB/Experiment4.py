import os
import sys
import glob
import cv2
import torch
from tqdm import tqdm
from detectron2.evaluation import COCOEvaluator
import numpy as np

import json
import matplotlib.pyplot as plt

from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.utils.logger import setup_logger
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer

def load_json_arr(json_path):
    lines = []
    with open(json_path, 'r') as f:
        for line in f:
            lines.append(json.loads(line))
    return lines

setup_logger()

file_dir = os.path.dirname(__file__)
print(file_dir)
sys.path.append(file_dir)

from KITTIMOTSLoader import get_KITTIMOTS_dicts
from vKITTILoader import get_vKITTI_dicts
import MaskConfiguration as mk
from ValidationTrainer import ValidationTrainer

#Args setup
if len(sys.argv) < 2:
    print('Syntax error, missing real data ratio, correct syntax is:')
    print('python3 Experiment4 ratio (-s)')

ratio = float(sys.argv[1])
bySeqs = True if '-s' in sys.argv else False

conf = "R50-FPN" # "R50-FPN-CS"

if bySeqs is True:
    ration = int(ratio)
    experiment = "exp4-{}-seqs".format(ratio)
else:
    experiment = "exp4-{}".format(int(ratio*100))

print("Base model: " + str(conf))

configuration = mk.MaskConfiguration().get_Configuration(conf)
PATH_RESULTS = './Results-{}-{}/'.format(conf, experiment)
os.makedirs(PATH_RESULTS, exist_ok=True)

print("////////////////////////////////////////////////////////")
print("////////////////////Configuration///////////////////////")
print("////////////////////////////////////////////////////////")
print(configuration[0])
print(configuration[1])
print("////////////////////////////////////////////////////////")
print("////////////////////////////////////////////////////////")
print("////////////////////////////////////////////////////////")

seqSets = [
    ("0001","0002","0006","0018","0020"),
    ("0000","0003","0010","0012","0014"),
    ("0004","0005","0007","0008","0009","0011","0015")
]

trainSet = 0
valSet = 1
testSet = 2

DatasetCatalog.register("kitti-train1", lambda: get_KITTIMOTS_dicts(seqSets[trainSet], ratio=ratio, bySeqs=bySeqs))
MetadataCatalog.get("kitti-train1").set(thing_classes=['Car', 'Pedestrian'])
trainDataset = get_KITTIMOTS_dicts(seqSets[trainSet], ratio=ratio, bySeqs=bySeqs)
trainDataCount = len(trainDataset)

DatasetCatalog.register("kitti-train2", lambda: get_vKITTI_dicts())
MetadataCatalog.get("kitti-train2").set(thing_classes=['Car', 'Pedestrian'])
trainDataset2 = get_vKITTI_dicts()
trainDataCount += len(trainDataset2)

DatasetCatalog.register("kitti-val", lambda: get_KITTIMOTS_dicts(seqSets[valSet]))
MetadataCatalog.get("kitti-val").set(thing_classes=['Car', 'Pedestrian'])

DatasetCatalog.register("kitti-test", lambda: get_KITTIMOTS_dicts(seqSets[testSet]))
MetadataCatalog.get("kitti-test").set(thing_classes=['Car', 'Pedestrian'])

cfg = get_cfg()

cfg.merge_from_file(configuration[0])
metadata = MetadataCatalog.get("kitti-train")
cfg.DATASETS.TRAIN = ("kitti-train",)
cfg.DATASETS.TEST = ('kitti-val',)
cfg.TEST.EVAL_PERIOD = 1000
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model

cfg.OUTPUT_DIR = PATH_RESULTS
#cfg.MODEL.WEIGHTS = configuration[1]

#Train parameters
cfg.DATALOADER.NUM_WORKERS = 2
cfg.SOLVER.IMS_PER_BATCH = 4
cfg.SOLVER.BASE_LR = 0.00025
itersToFullDataset = trainDataCount // cfg.SOLVER.IMS_PER_BATCH + 1
itersMultiplier = 5
itersDivider = 1
cfg.SOLVER.MAX_ITER = itersMultiplier * itersToFullDataset // itersDivider

cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = ValidationTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set the testing threshold for this model

experiment_metrics = load_json_arr(PATH_RESULTS + '/metrics.json')

plt.plot(
    [x['iteration'] for x in experiment_metrics], 
    [x['total_loss'] for x in experiment_metrics])
plt.plot(
    [x['iteration'] for x in experiment_metrics if 'validation_loss' in x], 
    [x['validation_loss'] for x in experiment_metrics if 'validation_loss' in x])
plt.legend(['total_loss', 'validation_loss'], loc='upper left')
plt.savefig(PATH_RESULTS + "/loss_plot_train.png")

# Set training data-set path
cfg.DATASETS.TEST = ('kitti-test',)

# Evaluation
evaluator = COCOEvaluator('kitti-test', cfg, False, output_dir='./output-{}'.format(experiment))
trainer.test(cfg, trainer.model, evaluators=[evaluator])

print("Generating images with predictions...")

datasetTest = get_KITTIMOTS_dicts(seqSets[testSet])

imagesToPredict = [97, 356, 527, 843, 1293, 1875, 2121, 2503]

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
predictor = DefaultPredictor(cfg)

for img_idx in imagesToPredict:
    filePath = datasetTest[img_idx]['file_name']
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