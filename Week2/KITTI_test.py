from detectron2.utils.visualizer import ColorMode
import random
import cv2
from kitti_load import get_KITTI_dicts
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
import os

#Create custom datasets
image_dir = 'datasets/KITTI/data_object_image_2/'
for dataset in ['mini_train', 'testing', 'training']:
    DatasetCatalog.register("KITTI/" + dataset, lambda dataset=dataset: get_KITTI_dicts(image_dir, dataset))
    MetadataCatalog.get("KITTI" + dataset).set(thing_classes=['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc', 'DontCare'])
print(MetadataCatalog.get('KITTI/mini_train'))

#Train the dataset
cfg = get_cfg()
cfg.merge_from_file("./detectron2_repo/configs/COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
cfg.DATASETS.TRAIN = ('KITTI/mini_train',)
cfg.DATASETS.TEST = ('KITTI/testing',)   # no metrics implemented for this dataset
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = "detectron2://COCO-Detection/faster_rcnn_R_101_FPN_3x/137851257/model_final_f6e8b1.pkl"
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.MAX_ITER = 10
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()

exit()
for d in random.sample(dataset_dicts, 3):
    im = cv2.imread(d["file_name"])

    cv2.imwrite('savedImage.jpg', im)