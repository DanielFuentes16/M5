import os
import glob
import cv2
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
from detectron2.structures import BoxMode
import mask
import pickle


def get_KITTIMOTS_dicts(set_type):
    if set_type is not 'train' and set_type is not 'test' and set_type is not 'val':
        raise Exception("Invalid set type")

    #Since the split is always the same, save the results to pickles
    if os.path.exists('kittiMots.pkl'):
        print("Loading data from local pickle file")
        data = pickle.load(open("kittiMots.pkl", "rb" ))
        return data[0] if set_type is 'train' else data[1]

    image_path = '/home/mcv/datasets/KITTI-MOTS/training/image_02'
    label_path = '/home/mcv/datasets/KITTI-MOTS/instances_txt'
    image_files = glob.glob(image_path + '/*/*.png')

    dataset_dicts = []

    for i, imageFile in tqdm(enumerate(image_files), total=len(image_files)):
        record = {}
        record["file_name"] = imageFile
        record["image_id"] = i

        splitPath = imageFile.split(os.sep)
        imageSet = splitPath[7]
        imageNum = int(splitPath[-1][:-4])
        labelFilename = label_path + '/' + imageSet + '.txt'

        objs = []
        with open(labelFilename) as f:
            lines = f.readlines()
        if len(lines) is 0:
            print('No lines in file')
            exit()

        height = record["height"] = lines[0][3]
        width = record["width"] = lines[0][4]

        for line in lines:
            col = line.split()
            if imageNum is col[0]:
                catg = int(col[1]) // 1000
                rle = {
                    'counts': col[5].strip(),
                    'size': [height, width]
                }
                obj = {
                    "bbox": mask.toBbox(rle),
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "category_id": catg
                }
                objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)

    trainData, valData, _, _ = train_test_split(dataset_dicts, dataset_dicts, test_size=0.20, random_state=42)
    pickle.dump([trainData, valData], open("kittiMots.pkl", "wb" ))
    return trainData if set_type is 'train' else valData


print(len(get_KITTIMOTS_dicts('train')))
print(len(get_KITTIMOTS_dicts('test')))