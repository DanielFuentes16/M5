import glob
import os
import cv2
from detectron2.structures import BoxMode

def get_KITTI_dicts(set_type):
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

    if set_type is 'mini_train':
        working_folder = '/home/mcv/datasets/KITTI/data_object_image_2/mini_train/'    
    if set_type is 'training':
        working_folder = '/home/mcv/datasets/KITTI/data_object_image_2/training/image_2'    
    if set_type is 'testing':
        working_folder = '/home/mcv/datasets/KITTI/data_object_image_2/testing/image_2'    
    
    #obtain
    #image_dir='/home/mcv/datasets/KITTI/data_object_image_2/training/image_2'
    #image_dir='/Users/danielfuentes/Desktop/KITTI/data_object_image_2/mini_train'
    #label_dir='/Users/danielfuentes/Desktop/KITTI/training/label_2'
    label_dir='/home/mcv/datasets/KITTI/training/label_2'
    image_path = glob.glob(working_folder + '/*.png')
    label_path = glob.glob(label_dir + '/*.txt')
    label_file = sorted(label_path)
    image_file = sorted(image_path)
    for file in image_file :
        splitd = file.split(os.sep)
        img_name = splitd[-1]
        img_id = img_name.split('.')[0]
        if set_type is not 'testing':
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

        if set_type is not 'testing':
            objs = []
            with open(label_file[i]) as f:
                lines = f.readlines()
            if len(lines) is 0:
                print('No lines in file')
                exit()
            for line in lines:
                col = line.split()
                catg = categories[col[0]]
                obj = {
                    "bbox": [float(col[4]), float(col[5]), float(col[6]), float(col[7])],
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "category_id": catg
                }
                objs.append(obj)
            record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts
