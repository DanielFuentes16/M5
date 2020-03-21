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
import MaskConfiguration as mk

inference = True
class MaskCNN_MOTS(object):
    def run(self, argv):

        #conf = "R50-C4"
        if len(argv) == 1:
            #configuration = mk.MaskConfiguration().get_Configuration(conf)
            exit(0)
        else:
            conf = argv[1]
            configuration = mk.MaskConfiguration().get_Configuration(conf)
        PATH_RESULTS = './Results-{}/'.format(conf)
        PATH_TRAIN = '/home/mcv/datasets/KITTI-MOTS/training/image_02/'
        PATH_TEST = '/home/mcv/datasets/KITTI/data_object_image_2/testing/image_2'
        os.makedirs(PATH_RESULTS, exist_ok=True)
        print("////////////////////////////////////////////////////////")
        print("////////////////////Configuration///////////////////////")
        print("////////////////////////////////////////////////////////")
        print(configuration[0])
        print(configuration[1])
        print("////////////////////////////////////////////////////////")
        print("////////////////////////////////////////////////////////")
        print("////////////////////////////////////////////////////////")
        for d in ["train", "val"]:
            DatasetCatalog.register("fcnn-mots" + d, lambda d=d: get_KITTIMOTS_dicts(d))
            MetadataCatalog.get("fcnn-mots" + d).set(thing_classes=['Car', 'DontCare', 'Pedestrian'])
        # Inference
        cfg = get_cfg()

        cfg.merge_from_file(configuration[0])
        #cfg.merge_from_file("./detectron2_repo/configs/COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
        metasata =  MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
        cfg.DATASETS.TRAIN = ('fcnn-motstrain',)
        cfg.DATASETS.TEST = ('fcnn-motsval',)
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model

        cfg.OUTPUT_DIR = PATH_RESULTS
        #cfg.MODEL.WEIGHTS = "detectron2://COCO-Detection/faster_rcnn_R_101_FPN_3x/137851257/model_final_f6e8b1.pkl"
        cfg.MODEL.WEIGHTS = configuration[1]

        if(inference):
            predictor = DefaultPredictor(cfg)
            for filePath in glob.glob(PATH_TRAIN + '/*/*.png'):
                path, filename = os.path.split(filePath)
                print(filePath)
                # Make prediction
                im = cv2.imread(filePath)
                outputs = predictor(im)

                # Visualize the prediction in the image
                #v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=0.8)
                v = Visualizer(
                    im[:, :, ::-1],
                    metadata=metasata,
                    scale=0.8,
                    instance_mode=ColorMode.IMAGE)
                v = v.draw_instance_predictions(outputs["instances"].to("cpu"))

                os.makedirs(PATH_RESULTS, exist_ok=True)
                cv2.imwrite(PATH_RESULTS + filename, v.get_image()[:, :, ::-1])

        #dataset_dicts = get_KITTIMOTS_dicts('val')


        # Inference
        #predictor = DefaultPredictor(cfg)
        print("Generating Predictions")
        predictions = []
       # for item in tqdm(dataset_dicts):
       #     # Make prediction
       #     im = cv2.imread(item['file_name'])
       #     outputs = predictor(im)
       #     cls_arr = outputs['instances']._fields['pred_classes'].cpu().numpy().tolist()
       #     scores_arr = outputs['instances']._fields['scores'].cpu().numpy().tolist()
       #     boxes_arr = outputs['instances']._fields['pred_boxes'].tensor.cpu().numpy().tolist()
       #     tmp_cls = []
       #     tmp_scores = []
       #     tmp_boxes = []
       #     for i, pack in enumerate(zip(cls_arr, scores_arr, boxes_arr)):
       #         cls_id = pack[0]
       #         score = pack[1]
       #         box = pack[2]
       #         if cls_id == 0:
       #             tmp_cls.append(1)
       #             tmp_scores.append(score)
       #             tmp_boxes.append(box)
       #         elif cls_id == 2:
       #             tmp_cls.append(0)
       #             tmp_scores.append(score)
       #             tmp_boxes.append(box)
       #     outputs['instances']._fields['pred_classes'] = torch.from_numpy(np.array(tmp_cls))
       #     outputs['instances']._fields['scores'] = torch.from_numpy(np.array(tmp_scores))
       #     outputs['instances']._fields['pred_boxes'].tensor = torch.from_numpy(np.array(tmp_boxes))
       #     predictions.append(outputs)

        model = build_model(cfg)
        DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)

        # Evaluation
        evaluator = COCOEvaluator('fcnn-motsval', cfg, False, output_dir='./output{}'.format(conf))
        trainer = DefaultTrainer(cfg)
        trainer.test(cfg, model, evaluators=[evaluator])

if __name__ == '__main__':
    MaskCNN_MOTS().run(sys.argv)


