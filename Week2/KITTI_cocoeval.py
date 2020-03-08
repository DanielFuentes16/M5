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
PATH_RESULTS = './Results/'


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

def get_KITTI_line(instance, idx):
    categories = [
        'Car',
        'Van',
        'Truck',
        'Pedestrian',
        'Person_sitting',
        'Cyclist',
        'Tram',
        'Misc',
        'DontCare']
    box = instance['pred_boxes'][idx].tensor.numpy()[0]
    score = instance['scores'][idx].numpy()
    class_idx = instance['pred_classes'][idx].numpy()
    string = str(categories[class_idx]) + " 0.00 0.00 0.00 " + str(box[0]) + " " + str(box[1]) + " " + str(box[2]) + " " + str(box[3]) + " 0.00 0.00 0.00 0.00 0.00 0.00 0.00 " + str(score) 
    return string
    #return "Car 1.00 0 2.50 0.00 209.77 127.87 374.00 1.62 1.75 4.30 -4.15 1.85 2.70 1.55"

for d in ["train", "val"]:
    DatasetCatalog.register("kitti" + d, lambda d=d: get_KITTI_dicts('training' if d == 'train' else 'testing'))
    MetadataCatalog.get("kitti" + d).set(thing_classes=['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc', 'DontCare'])
kitti_metadata = MetadataCatalog.get("kittitrain")

cfg = get_cfg()
cfg.merge_from_file("./detectron2_repo/configs/COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
cfg.DATASETS.TRAIN = ("kittitrain",)
cfg.DATASETS.TEST = ()   # no metrics implemented for this dataset
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = "detectron2://COCO-Detection/faster_rcnn_R_101_FPN_3x/137851257/model_final_f6e8b1.pkl"
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.MAX_ITER = 500
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
exit()

# Create predictor (model for inference)
predictor = DefaultPredictor(cfg)

os.makedirs(PATH_RESULTS + 'data/', exist_ok=True)
dataset_dicts = get_KITTI_dicts('testing')
for d in tqdm(dataset_dicts):
    im = cv2.imread(d["file_name"])
    _, filename = os.path.split(d["file_name"])
    outputs = predictor(im)
    with open(PATH_RESULTS + 'data/' + filename[0:-4] + '.txt', 'w') as f:
        instance = outputs["instances"].to("cpu")
        for idx in range(len(instance)):
            f.write(get_KITTI_line(instance._fields, idx) + "\n")
    #v = Visualizer(im[:, :, ::-1], metadata=kitti_metadata, scale=0.8)
    #v = v.draw_instance_predictions(outputs["instances"].to("cpu"))

    #os.makedirs(PATH_RESULTS, exist_ok=True)
    #cv2.imwrite(PATH_RESULTS + filename, v.get_image()[:, :, ::-1])
