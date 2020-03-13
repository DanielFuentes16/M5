import glob
import os
import cv2
from detectron2.structures import BoxMode
from pycocotools import coco


categories = {
    'Car': 1,
    'Pedestrian': 2,
    }
    
KITTI_MOTS_train='/home/mcv/datasets/KITTI-MOTS/training/image_02'
KITTI_MOTS_test ='/home/mcv/datasets/KITTI-MOTS/testing/image_02'
KITTI_MOTS_label = '/home/mcv/datasets/KITTI-MOTS/instances_txt' 

def get_kittimots_dicts(KITTI_MOTS_train,KITTI_MOTS_label,train_perc=0.75,train):
    os.path.isdir(KITTI_MOTS_train)
    os.path.isdir(KITTI_MOTS_label)
    label_path = sorted(glob(KITTI_MOTS_label+os.sep+'*.txt'))
    label = [format(l) for l in range(len(label_path))]
    n_train_seq = int(len(img)*split_perc)
    train_seq = label[:n_train_seq]
    test_seq = label[n_train_seq:]
    
    sequences = train_seq if train else test_seq
    
    kitti_mots_dicts=[]
    for seq in sequences:
        seq_dicts = []
        seq_img_path = sorted(glob(KITTI_MOTS_train+os.sep+seq+os.sep+'*.png'))
        n_frames = len(seq_img_path)
        seq_label_path = KITTI_MOTS_labelL+os.sep+seq+'.txt'
        
        with open(seq_label_path,'r') as f:
            lines = file.readlines()
            lines = [l.split(' ') for l in lines]
            
        for frame in range(n_frames):
            frame_lines = [l for l in lines if int(l[0]) == frame]
            frame_annot = []
            
            h, w = int(frame_lines[0][3]), int(frame_lines[0][4])
            
            for line in frame_lines:
                seg = {
                    'counts': line[-1].strip(),
                    'size': [h, w]
                }
                
                box = coco.maskUtils.toBbox(seg)
                
                annot = {
                    'category_id': int(line[1])%1000,
                    'bbox_mode':BoxMode.XYXY_ABS,
                    'bbox':box
                }
                frame_annot.append(annot)
                
            img_dict = {
                'file_name': os.path.join(KITTI_MOTS_train,seq,format(frame)),
                'image_id': frame+(int(seq)*1e3),
                'height': h,
                'width': w,
                'annotations': frame_annot
            }
            seq_dicts.append(img_dict)

    return seq_dicts
