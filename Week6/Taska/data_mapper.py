#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 20:27:09 2020

@author: kaiali
"""
import copy
import torch
import numpy as np
from fvcore.common.file_io import PathManager
from PIL import Image

from detectron2.data import build_detection_train_loader
from detectron2.data import transforms as T
from detectron2.data import detection_utils as utils

def dataset_mapper(dataset_dict, cfg, is_train=True ):   
    # Implement a mapper, similar to the default DatasetMapper, but with your own customizations
    dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
    image = utils.read_image(dataset_dict["file_name"], format="BGR")
    #image, transforms = T.apply_transform_gens([T.Resize((800, 800))], image)
    #dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))
    utils.check_image_size(dataset_dict, image)
    
    if is_train:
        min_size = cfg.INPUT.MIN_SIZE_TRAIN
        max_size = cfg.INPUT.MAX_SIZE_TRAIN
        flip_horizontal_prob = cfg.INPUT.HORIZONTAL_FLIP_PROB_TRAIN
        flip_vertical_prob = cfg.INPUT.VERTICAL_FLIP_PROB_TRAIN
        contrast = cfg.INPUT.CONTRAST
    else:
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        flip_horizontal_prob = 0.0
        flip_vertical_prob = 0.0
        contrast = 0.0
        
    norm_transform = T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    
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

        annos = [utils.transform_instance_annotations(obj, transforms, image_shape, keypoint_hflip_indices=None) for obj
                 in dataset_dict.pop("annotations") if obj.get("iscrowd", 0) == 0]

        instances = utils.annotations_to_instances(annos, image_shape)
        # Create a tight bounding box from masks, useful when image is cropped
        if crop_transform is not None and instances.has("gt_masks"):
            instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
        dataset_dict["instances"] = utils.filter_empty_instances(instances)
        
    return dataset_dict

if __name__ == '__main__':
    data_loader = build_detection_train_loader(dataset_mapper)

   