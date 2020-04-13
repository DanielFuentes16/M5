import os
import glob
import cv2
from tqdm import tqdm
from pycocotools import coco
from detectron2.structures import BoxMode
import mask
import pickle
import hashlib
from sklearn.model_selection import train_test_split


def get_KITTIMOTS_dicts(seqs, useCache=True, ratio=1.0, bySeqs=False):

    pkl_name = hashlib.md5(''.join(seqs).encode()).hexdigest() + '.pkl'

    if ratio < 1.0:
        useCache = False

    if useCache is True and os.path.exists(pkl_name):
        print("Loading data from local pickle file")
        data = pickle.load(open(pkl_name, "rb"))
        return data

    image_path = '/home/mcv/datasets/KITTI-MOTS/training/image_02'
    label_path = '/home/mcv/datasets/KITTI-MOTS/instances_txt'

    train_seqs = ["0000","0001","0003","0004","0005","0009","0011","0012","0015","0017","0019","0020"]
    val_seqs = ["0002","0006","0007","0008","0010","0013","0014","0016","0018"]

    dataset_dicts = []

    i=0 
    for seq in seqs:
        path_regex = image_path + '/' + seq + '/*.png'
        imgPaths = glob.glob(path_regex)
        imgPaths = sorted(imgPaths)
        if bySeqs is True:
            nImages = int(len(imgPaths) * ratio)
            imgPaths = imgPaths[0:nImages]
        for imageFile in tqdm(imgPaths):
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
            i += 1
    
    if bySeqs is False:
        dataset_dicts, _ = train_test_split(dataset_dicts, train_size=ratio)

    if useCache is True:
        pickle.dump(dataset_dicts, open(pkl_name, "wb"))
    return dataset_dicts