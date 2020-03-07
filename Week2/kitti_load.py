import glob
import os
import cv2
from detectron2.structures import BoxMode

def get_KITTI_dicts(image_dir):
    categories = {
    'Car': 0,
    'Van': 1,
    'Truck': 2,
    'Pedestrian': 3,
    'Person_sitting': 4,
    'Cyclist': 5,
    'Tram': 6,
    'Misc': 7,
    'DontCare': 8
    }
    
    #obtain
    #image_dir='/home/mcv/datasets/KITTI/data_object_image_2/training/image_2'
    #image_dir='/Users/danielfuentes/Desktop/KITTI/data_object_image_2/mini_train'
    #label_dir='/Users/danielfuentes/Desktop/KITTI/training/label_2'
    label_dir='/home/mcv/datasets/KITTI/training/label_2'
    image_path = glob.glob(image_dir+ '/*.png')
    label_path = glob.glob(label_dir + '/*.txt')
    label_file = sorted(label_path)
    image_file = sorted(image_path)
    for file in image_file :
        splitd = file.split(os.sep)
        img_name = splitd[-1]
        img_id = img_name.split('.')[0]
        label_file.append(label_dir + img_id + '.txt')

    dataset_dicts = []
    
    #iteration
    for i in range(0,len(image_file)):
        record = {}
        height, width = cv2.imread(image_file[i]).shape[:2]
        record["file_name"] = image_file[i]
        record["image_id"] = i
        record["height"] = height
        record["width"] = width

        objs = []
        with open(label_file[i]) as f:
            lines = f.readlines()   
        for line in lines:
            col = line.split()
            catg = categories[col[0]]
            obj = {
                "bbox": [col[4], col[5], col[6], col[7]],
                "bbox_mode": BoxMode.XYXY_ABS,
                "category_id": catg
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return(dataset_dicts)
