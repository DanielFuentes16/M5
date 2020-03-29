import os
import sys
import glob
import cv2
import pickle
from tqdm import tqdm

from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.utils.logger import setup_logger
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
setup_logger()

file_dir = os.path.dirname(__file__)
print(file_dir)
sys.path.append(file_dir)

from KITTIMOTSLoader import get_KITTIMOTS_dicts
import MaskConfiguration as mk

inference = False

class MaskCNN_MOTS(object):
    def run(self, argv):

        if len(argv) == 1:
            exit(0)
        else:
            conf = argv[1]
            configuration = mk.MaskConfiguration().get_Configuration(conf)
            check = '16'
            checkpoint = configuration[1]
        if len(argv) == 3:
            check = argv[2]
            checkpoint = mk.MaskConfiguration().get_Checkpoint(check)

        PATH_RESULTS = './Results-{}-{}/'.format(conf, check)
        PATH_TRAIN = '/home/mcv/datasets/KITTI-MOTS/training/image_02/'
        os.makedirs(PATH_RESULTS, exist_ok=True)

        print("////////////////////////////////////////////////////////")
        print("////////////////////Configuration///////////////////////")
        print("////////////////////////////////////////////////////////")
        print(configuration[0])
        print(configuration[1])
        print("////////////////////////////////////////////////////////")
        print("////////////////////////////////////////////////////////")
        print("////////////////////////////////////////////////////////")

        #data = pickle.load(open("preds.pkl", "rb"))
        #data_pre = pickle.load(open("preds_pre.pkl", "rb"))

        if len(sys.argv) == 3:
            useCOCO = False
            classes = ['Car', 'Pedestrian']
        else:
            useCOCO = True
            classes = ['person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle']

        for d in ["val"]:
            DatasetCatalog.register("fcnn-mots" + d, lambda d=d: get_KITTIMOTS_dicts(d, True))
            MetadataCatalog.get("fcnn-mots" + d).set(thing_classes=classes)
        
        cfg = get_cfg()
        cfg.merge_from_file(configuration[0])
        metasata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0] if useCOCO is True else "fcnn-motsfull")
        cfg.DATASETS.TEST = ('fcnn-motsval',)
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        cfg.OUTPUT_DIR = PATH_RESULTS
        # cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
        cfg.MODEL.WEIGHTS = checkpoint

        if(inference):
            predictor = DefaultPredictor(cfg)
            for filePath in glob.glob(PATH_TRAIN + '/*/*.png'):
                path, filename = os.path.split(filePath)
                print(filePath)
                # Make prediction
                im = cv2.imread(filePath)
                outputs = predictor(im)
                v = Visualizer(
                    im[:, :, ::-1],
                    metadata=metasata,
                    scale=0.8,
                    instance_mode=ColorMode.IMAGE)
                v = v.draw_instance_predictions(outputs["instances"].to("cpu"))

                os.makedirs(PATH_RESULTS, exist_ok=True)
                cv2.imwrite(PATH_RESULTS + filename, v.get_image()[:, :, ::-1])

        print("Generating Predictions")
        model = build_model(cfg)
        DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)

        # Evaluation
        #transform_dict = {2: 0, 0: 1,}
        transform_dict = {0: 1, 2: 0}
        evaluator = COCOEvaluator('fcnn-motsval', cfg, False, output_dir='./output{}-{}/'.format(conf, check))
        val_loader = build_detection_test_loader(cfg, 'fcnn-motsval')
        inference_on_dataset(model, val_loader, evaluator)

if __name__ == '__main__':
    MaskCNN_MOTS().run(sys.argv)


