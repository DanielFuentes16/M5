import cv2
import glob
import os
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.logger import setup_logger

import time

setup_logger()

Debug = False

PATH_TRAIN = '/home/mcv/datasets/KITTI-MOTS/training/image_02/'
PATH_TEST = '/home/mcv/datasets/KITTI/data_object_image_2/testing/image_2'
PATH_RESULTS = './ResultsRetinaNetR101/'

# Create config
cfg = get_cfg()
cfg.merge_from_file("../Week2/detectron2_repo/configs/COCO-Detection/retinanet_R_101_FPN_3x.yaml")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set threshold for this model
cfg.MODEL.WEIGHTS = "detectron2://COCO-Detection/retinanet_R_101_FPN_3x/138363263/model_final_59f53c.pkl"

# Inference
predictor = DefaultPredictor(cfg)
tic = time.perf_counter()
count = 0;
counter = 0;
for filePath in glob.glob(PATH_TRAIN + '/*/*.png'):
    count = count + 1
    tic = time.perf_counter()
    path, filename = os.path.split(filePath)

    if Debug:
        print(filePath)

    # Make prediction
    im = cv2.imread(filePath)
    outputs = predictor(im)

    # Visualize the prediction in the image
    v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))

    if Debug:
        print(filename)
    toc = time.perf_counter()
    counter = counter + (toc - tic)
    os.makedirs(PATH_RESULTS, exist_ok=True)
    cv2.imwrite(PATH_RESULTS + filename, v.get_image()[:, :, ::-1])

print(f"Downloaded the tutorial in {counter/count:0.4f} (s/im)")
