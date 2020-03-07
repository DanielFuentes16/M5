import cv2
import glob
import os
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.utils.logger import setup_logger
setup_logger()

Debug = False

PATH_IMAGES = '../datasets/MIT_split/train/'
PATH_RESULTS = './Results/'
SUB_FOLDERS = ['coast', 'forest', 'highway', 'inside_city', 'mountain', 'Opencountry', 'street', 'tallbuilding']

# Create config
cfg = get_cfg()
cfg.merge_from_file("./detectron2_repo/configs/COCO-Detection/retinanet_R_101_FPN_3x.yaml")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
cfg.MODEL.WEIGHTS = "detectron2://COCO-Detection/retinanet_R_101_FPN_3x/138363263/model_final_59f53c.pkl"

# Create predictor
predictor = DefaultPredictor(cfg)

for sub_folder in SUB_FOLDERS:
    for filePath in glob.glob(PATH_IMAGES + sub_folder + '/*.jpg'):
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

        os.makedirs(PATH_RESULTS, exist_ok=True)
        cv2.imwrite(PATH_RESULTS + filename, v.get_image()[:, :, ::-1])
