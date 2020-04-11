import os
import glob
import cv2
from tqdm import tqdm
from pycocotools import coco
from detectron2.structures import BoxMode
import mask
import pickle


def get_KITTIMOTS_dicts(set_type):
    if set_type is not 'train' and set_type is not 'test' and set_type is not 'val':
        raise Exception("Invalid set type")

    pkl_name = 'kittiMots_train.pkl' if set_type is "train" else 'kittiMots_val.pkl'

    if os.path.exists(pkl_name):
        print("Loading data from local pickle file")
        data = pickle.load(open(pkl_name, "rb"))
        return data

    image_path = '/home/mcv/datasets/KITTI-MOTS/training/image_02'
    label_path = '/home/mcv/datasets/KITTI-MOTS/instances_txt'

    train_seqs = ["0000","0001","0003","0004","0005","0009","0011","0012","0015","0017","0019","0020"]
    val_seqs = ["0002","0006","0007","0008","0010","0013","0014","0016","0018"]

    seqs = train_seqs if set_type is "train" else val_seqs

    image_files = []

    for seq in seqs:
        path_regex = image_path + '/' + seq + '/*.png'
        for path in glob.glob(path_regex):
            image_files.append(path)

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

        height = record["height"] = int(lines[0].split()[3])
        width = record["width"] = int(lines[0].split()[4])

        for line in lines:
            col = line.split()
            if imageNum is int(col[0]):
                catg = int(col[1]) // 1000
                if catg is 1:
                    catg = 0
                elif catg is 2:
                    catg = 1
                else:
                    continue
                    catg = 2
                rle = {
                    'counts': col[5].strip(),
                    'size': [height, width]
                }
                bbox = mask.toBbox(rle)
                maskk = coco.maskUtils.decode(rle)
                contours, _ = cv2.findContours(maskk, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                seg = [[int(i) for i in c.flatten()] for c in contours]
                seg = [s for s in seg if len(s) >= 6]
                if not seg:
                    continue

                obj = {
                    "bbox": (bbox[0], bbox[1], bbox[2], bbox[3]),
                    "bbox_mode": BoxMode.XYWH_ABS,
                    "category_id": catg,
                    'segmentation': seg
                }
                objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)

    pickle.dump(dataset_dicts, open(pkl_name, "wb"))
    return dataset_dicts