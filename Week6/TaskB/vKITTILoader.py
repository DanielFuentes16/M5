import os
import glob
import cv2
from tqdm import tqdm
from pycocotools import coco
from detectron2.structures import BoxMode
import mask
import pickle
import hashlib
import numpy as np
import pycocotools.mask as mask_utils
from itertools import groupby

from skimage.measure import regionprops

def binary_mask_to_rle(binary_mask):
    rle = {'counts': [], 'size': list(binary_mask.shape)}
    counts = rle.get('counts')
    for i, (value, elements) in enumerate(groupby(binary_mask.ravel(order='F'))):
        if i == 0 and value == 1:
            counts.append(0)
        counts.append(len(list(elements)))
    return rle

def get_vKITTI_dicts(useCache=True):
    pkl_name = 'vKITTI.pkl'

    if useCache is True and os.path.exists(pkl_name):
        print("Loading data from local pickle file")
        data = pickle.load(open(pkl_name, "rb"))
        return data

    basePath = '/home/mcv/datasets/vKITTI/'
    scenes = ('Scene01', 'Scene02', 'Scene06', 'Scene18', 'Scene20')
    imageSuffix = '/clone/frames/rgb/'
    maskSuffix = '/clone/frames/instanceSegmentation/'

    image_files = []
    dataset_dicts = []

    i = 0
    for scene in tqdm(scenes):
        path_regex = basePath + scene + imageSuffix + '*/*.jpg'
        for imageFile in tqdm(glob.glob(path_regex)):
            record = {}
            record["file_name"] = imageFile
            record["image_id"] = i
            record["scene"] = scene

            splitPath = imageFile.split(os.sep)
            scene = splitPath[-6]
            camera = splitPath[-2]
            imgNum = splitPath[-1][4:-4]
            labelFilename = basePath + scene + maskSuffix + camera + '/instancegt_' + imgNum + '.png'

            img = cv2.imread(labelFilename, cv2.IMREAD_GRAYSCALE)
            h, w = img.shape
            frame_annotations = []
            instances = np.unique(img)
            for ins in instances[1:]:
                mask = np.copy(img)
                mask[(mask==ins)] = 1
                mask[(mask!=1)] = 0
                props = regionprops(mask)
                tmp_bbox = props[0].bbox
                bbox = list(tmp_bbox)
                bbox[0] = tmp_bbox[1]
                bbox[1] = tmp_bbox[0]
                bbox[2] = tmp_bbox[3]
                bbox[3] = tmp_bbox[2]
                contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                seg = [[int(i) for i in c.flatten()] for c in contours]
                seg = [s for s in seg if len(s) >= 6]
                if not seg:
                    continue
                annotation = {
                    'category_id': 0,
                    'bbox_mode': BoxMode.XYXY_ABS,
                    'bbox': bbox,
                    'segmentation': seg,
                }
                frame_annotations.append(annotation)
            record["annotations"] = frame_annotations
            dataset_dicts.append(record)
            i += 1

    if useCache is True:
        pickle.dump(dataset_dicts, open(pkl_name, "wb"))
    return dataset_dicts