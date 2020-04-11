
import copy
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
        self.da = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]).get('da')
        self.tfm_gens = []

    def __call__(self, dataset_dict):
        # Implement a mapper, similar to the default DatasetMapper, but with your own customizations
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        print(dataset_dict["file_name"])
        image = utils.read_image(dataset_dict["file_name"], format="BGR")
        # image, transforms = T.apply_transform_gens([T.Resize((800, 800))], image)
        # dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))
        utils.check_image_size(dataset_dict, image)

        if self.is_train:
            min_size = self.cfg.INPUT.MIN_SIZE_TRAIN
            max_size = self.cfg.INPUT.MAX_SIZE_TRAIN
            flip_horizontal_prob = self.cfg.INPUT.HORIZONTAL_FLIP_PROB_TRAIN
            flip_vertical_prob = self.cfg.INPUT.VERTICAL_FLIP_PROB_TRAIN
            contrast = self.cfg.INPUT.CONTRAST
        else:
            min_size = self.cfg.INPUT.MIN_SIZE_TEST
            max_size = self.cfg.INPUT.MAX_SIZE_TEST
            flip_horizontal_prob = 0.0
            flip_vertical_prob = 0.0
            contrast = 0.0

        norm_transform = T.Normalize(mean=self.cfg.INPUT.PIXEL_MEAN, std=self.cfg.INPUT.PIXEL_STD)

        color_jitter = T.ColorJitter(
            contrast=contrast,
        )

        transform = T.Compose(
            [
                T.Resize(min_size, max_size),
                T.RandomHorizontalFlip(flip_horizontal_prob),
                T.RandomVerticalFlip(flip_vertical_prob),
                T.ToTensor(),
                norm_transform,
            ]
        )

        image_shape = image.shape[:2]  # h, w

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        crop_transform = None
        transforms = None

        if "annotations" in dataset_dict:

            annos = [utils.transform_instance_annotations(obj, transforms, image_shape, keypoint_hflip_indices=None) for
                     obj
                     in dataset_dict.pop("annotations") if obj.get("iscrowd", 0) == 0]

            instances = utils.annotations_to_instances(annos, image_shape)
            # Create a tight bounding box from masks, useful when image is cropped
            if crop_transform is not None and instances.has("gt_masks"):
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            dataset_dict["instances"] = utils.filter_empty_instances(instances)

        return dataset_dict

