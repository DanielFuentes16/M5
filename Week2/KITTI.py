
import cv2
import glob
import random
from detectron2.utils.visualizer import Visualizer
from detectron2.engine import DefaultTrainer
from detectron2.data import DatasetCatalog
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
import os

PATH_TRAIN = '../datasets/MIT_split/train/'

# write a function that loads the dataset into detectron2's standard format
def get_microcontroller_dicts():
    classes = ['coast', 'forest', 'highway', 'inside_city', 'mountain', 'Opencountry', 'street', 'tallbuilding']
    dataset_dicts = []
    for class_i in classes:
        for filename in glob.glob(PATH_TRAIN+class_i+'/*.jpg'):
            record = {}

            height, width = cv2.imread(filename).shape[:2]

            record["file_name"] = filename
            record["height"] = height
            record["width"] = width

            objs = []
            obj = {'category_id': class_i}
            objs.append(obj)
            record["annotations"] = objs
            dataset_dicts.append(record)
            print(filename)
    return dataset_dicts

DatasetCatalog.register("my_dataset", get_microcontroller_dicts)
MetadataCatalog.get("my_dataset").thing_classes = ['coast', 'forest', 'highway', 'inside_city', 'mountain', 'Opencountry', 'street', 'tallbuilding']

cfg = get_cfg()
cfg.merge_from_file(
    "./detectron2_repo/configs/COCO-Detection/faster_rcnn_R_50_C4_1x.yaml"
)
#cfg.DATASETS.TRAIN = ("my_dataset",)
cfg.DATASETS.TEST = ()  # no metrics implemented for this dataset
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = "detectron2://ImageNetPretrained/MSRA/R-50.pkl"  #
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.02
cfg.SOLVER.MAX_ITER = (
    300
)  # 300 iterations seems good enough, but you can certainly train longer
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = (
    128
)  # faster, and good enough for this toy dataset
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 8  # 3 classes (data, fig, hazelnut)

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set the testing threshold for this model
cfg.DATASETS.TEST = ("fruits_nuts", )
predictor = DefaultPredictor(cfg)

from detectron2.utils.visualizer import ColorMode

for d in random.sample(dataset_dicts, 3):
    im = cv2.imread(d["file_name"])
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1],
                   metadata=fruits_nuts_metadata,
                   scale=0.8,
                   instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels
    )
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2_imshow(v.get_image()[:, :, ::-1])