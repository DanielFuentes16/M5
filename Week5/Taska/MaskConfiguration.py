class MaskConfiguration:
    def __init__(self):
        self.configurations = {
        "R50-C4" : ('/home/grupo09/df/Week2/detectron2_repo/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_C4_1x.yaml' ,
                          'detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_C4_1x/137259246/model_final_9243eb.pkl'),

        "R50-DC5" : ('/home/grupo09/df/Week2/detectron2_repo/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_DC5_1x.yaml',
                          'detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_DC5_1x/137260150/model_final_4f86c3.pkl'),

        "R50-FPN" : ('/home/grupo09/df/Week2/detectron2_repo/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml',
                          'detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x/137260431/model_final_a54504.pkl'),

        "R101-C4" : ('/home/grupo09/df/Week2/detectron2_repo/configs/COCO-InstanceSegmentation/mask_rcnn_R_101_C4_3x.yaml',
                          'detectron2://COCO-InstanceSegmentation/mask_rcnn_R_101_C4_3x/138363239/model_final_a2914c.pkl'),

        "R101-DC5" : ('/home/grupo09/df/Week2/detectron2_repo/configs/COCO-InstanceSegmentation/mask_rcnn_R_101_DC5_3x.yaml',
                          'detectron2://COCO-InstanceSegmentation/mask_rcnn_R_101_DC5_3x/138363294/model_final_0464b7.pkl'),

        "R101-FPN" : ('/home/grupo09/df/Week2/detectron2_repo/configs/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml',
                          'detectron2://COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x/138205316/model_final_a3ec72.pkl'),

        "R50-FPN-CS" : ('/home/grupo09/df/Week2/detectron2_repo/configs/Cityscapes/mask_rcnn_R_50_FPN.yaml',
                          'detectron2://Cityscapes/mask_rcnn_R_50_FPN/142423278/model_final_af9cf5.pkl')
        }

        self.checkpoints = {
            "Kitti": '/home/grupo09/df/Week5/Taska/Checkpoints/model_final.pth',
            "City": '/home/grupo09/df/Week5/Taska/Checkpoints/model_final_CS.pth'
        }
    def get_Configuration(self, configuration):
        return self.configurations.get(configuration)

    def get_Checkpoint(self, checkpoint):
        return self.checkpoints.get(checkpoint)
