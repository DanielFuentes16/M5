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
setup_logger('zutput_file')

PATH_TRAIN = '/home/mcv/datasets/KITTI/data_object_image_2/mini_train'
PATH_TEST = '/home/mcv/datasets/KITTI/data_object_image_2/testing/image_2'


def get_KITTI_dicts(set_type):
    categories = {
        'Car': 0,
        'Van': 1,
        'Truck': 2,
        'Pedestrian': 3,
        'Person_sitting': 4,
        'Cyclist': 5,
        'Tram': 6,
        'Misc': 7,
        'DontCare': 8
    }

    if set_type is 'mini_train':
        working_folder = '/home/mcv/datasets/KITTI/data_object_image_2/mini_train/'
    if set_type is 'training':
        working_folder = '/home/mcv/datasets/KITTI/data_object_image_2/training/image_2'
    if set_type is 'testing':
        working_folder = '/home/mcv/datasets/KITTI/data_object_image_2/testing/image_2'

    # obtain
    # image_dir='/home/mcv/datasets/KITTI/data_object_image_2/training/image_2'
    # image_dir='/Users/danielfuentes/Desktop/KITTI/data_object_image_2/mini_train'
    # label_dir='/Users/danielfuentes/Desktop/KITTI/training/label_2'
    label_dir = '/home/mcv/datasets/KITTI/training/label_2'
    image_path = glob.glob(working_folder + '/*.png')
    label_path = glob.glob(label_dir + '/*.txt')
    label_file = sorted(label_path)
    image_file = sorted(image_path)
    for file in image_file:
        splitd = file.split(os.sep)
        img_name = splitd[-1]
        img_id = img_name.split('.')[0]
        if set_type is not 'testing':
            label_file.append(label_dir + img_id + '.txt')

    dataset_dicts = []

    # iteration
    for i in range(0, len(image_file)):
        record = {}
        height, width = cv2.imread(image_file[i]).shape[:2]
        record["file_name"] = image_file[i]
        record["image_id"] = i
        record["height"] = height
        record["width"] = width

        if set_type is not 'testing':
            objs = []
            with open(label_file[i]) as f:
                lines = f.readlines()
            if len(lines) is 0:
                print('No lines in file')
                exit()
            for line in lines:
                col = line.split()
                catg = categories[col[0]]
                obj = {
                    "bbox": [float(col[4]), float(col[5]), float(col[6]), float(col[7])],
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "category_id": catg
                }
                objs.append(obj)
            record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts

for d in ["train", "val"]:
    DatasetCatalog.register("kitti" + d, lambda d=d: get_KITTI_dicts('training' if d == 'train' else 'testing'))
    MetadataCatalog.get("kitti" + d).set(thing_classes=['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc', 'DontCare'])
kitti_metadata = MetadataCatalog.get("kittitrain")

dataset_dicts = get_KITTI_dicts('testing')

cfg = get_cfg()
cfg.merge_from_file("./detectron2_repo/configs/COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
cfg.DATASETS.TRAIN = ("kittitrain",)
cfg.DATASETS.TEST = ()   # no metrics implemented for this dataset
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = "detectron2://COCO-Detection/faster_rcnn_R_101_FPN_3x/137851257/model_final_f6e8b1.pkl"
cfg.SOLVER.IMS_PER_BATCH = 4
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 5 * len(dataset_dicts) // cfg.SOLVER.IMS_PER_BATCH + 1 
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 9

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()

# load weights
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set the testing threshold for this model

# Set training data-set path
cfg.DATASETS.TEST = ("kittitrain", )

evaluator = COCOEvaluator("kittitrain", cfg, False, output_dir="./output/")
trainer.test(cfg, trainer.model, evaluators=[evaluator])
