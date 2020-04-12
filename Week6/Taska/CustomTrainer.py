
import torch
import numpy as np

from copy import deepcopy
from detectron2.engine import DefaultTrainer
from detectron2.data import build_detection_train_loader, build_detection_test_loader, DatasetMapper, MetadataCatalog
from detectron2.data import transforms as T
from detectron2.data import detection_utils as utils

class CustomTrainer(DefaultTrainer):

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        return build_detection_test_loader(cfg, dataset_name, mapper=CustomMapper(cfg, False))

    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=CustomMapper(cfg, True))


class CustomMapper(DatasetMapper):

    def __init__(self, cfg, is_train=True):
        super().__init__(cfg, is_train=is_train)
        self.cfg = deepcopy(cfg)
        self.generations = []
        self.methods = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]).get('methods')

    def __call__(self, dataset_dict):
        self.tfm_gens = []

        dataset_dict = deepcopy(dataset_dict)
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)

        if self.is_train:
            # Crop
            if 'crop' in self.methods.keys():
                crop_gen = T.RandomCrop(self.methods['crop']['type'], self.methods['crop']['size'])
                self.generations.append(crop_gen)
            # Horizontal flip
            if 'flip' in self.methods.keys():
                flip_gen = T.RandomFlip(prob=self.methods['flip']['prob'], horizontal=self.methods['flip']['horizontal'],
                                        vertical=self.methods['flip']['vertical'])
                self.generations.append(flip_gen)

        image, transforms = T.apply_transform_gens(self.generations, image)

        image_shape = image.shape[:2]  # h, w

        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

        if not self.is_train:
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            return dataset_dict

        if "annotations" in dataset_dict:

            annos = [utils.transform_instance_annotations(obj, transforms, image_shape, keypoint_hflip_indices=None) for
                     obj
                     in dataset_dict.pop("annotations") if obj.get("iscrowd", 0) == 0]

            instances = utils.annotations_to_instances(annos, image_shape)
            # Create a tight bounding box from masks, useful when image is cropped
            if self.crop_gen is not None and instances.has("gt_masks"):
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            dataset_dict["instances"] = utils.filter_empty_instances(instances)

        return dataset_dict
