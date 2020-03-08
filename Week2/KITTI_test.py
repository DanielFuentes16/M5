from detectron2.utils.visualizer import ColorMode
import random
import cv2
from kitti_load import get_KITTI_dicts
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
import os
from tqdm import tqdm

#Create custom datasets
image_dir = 'datasets/KITTI/data_object_image_2/'
PATH_RESULTS = './results/'
for dataset in ['mini_train', 'testing', 'training']:
    DatasetCatalog.register("KITTI/" + dataset, lambda dataset=dataset: get_KITTI_dicts(dataset))
    MetadataCatalog.get("KITTI" + dataset).set(thing_classes=['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc', 'DontCare'])
mini_metadata = MetadataCatalog.get('KITTI/mini_train')

#Train the dataset
cfg = get_cfg()
cfg.merge_from_file("./detectron2_repo/configs/COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
cfg.DATASETS.TRAIN = ('KITTI/training',)
cfg.DATASETS.TEST = ()   # no metrics implemented for this dataset
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = "detectron2://COCO-Detection/faster_rcnn_R_101_FPN_3x/137851257/model_final_f6e8b1.pkl"
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.MAX_ITER = 5000
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 9

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()

cfg = get_cfg()
cfg.merge_from_file("./detectron2_repo/configs/COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.1   # set the testing threshold for this model
cfg.DATASETS.TEST = ('KITTI/test', )
predictor = DefaultPredictor(cfg)

dataset_dicts = get_KITTI_dicts('testing')
print()
for d in tqdm(random.sample(dataset_dicts, 100)):
    im = cv2.imread(d["file_name"])
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1], metadata=mini_metadata, scale=0.8)
    if len(outputs['instances']) > 0:
        print('Found something')
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    os.makedirs(PATH_RESULTS, exist_ok=True)
    path, filename = os.path.split(d["file_name"])
    cv2.imwrite(PATH_RESULTS + filename, v.get_image()[:, :, ::-1])

exit()