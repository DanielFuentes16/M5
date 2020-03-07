from detectron2.utils.visualizer import ColorMode
import random
import cv2
from Week2.kitti_load import get_KITTI_dicts

#image_dir='Your path'
image_dir='/Users/danielfuentes/Desktop/KITTI/data_object_image_2/mini_train'
dataset_dicts = get_KITTI_dicts(image_dir)
for d in random.sample(dataset_dicts, 3):
    im = cv2.imread(d["file_name"])

    cv2.imwrite('savedImage.jpg', im)