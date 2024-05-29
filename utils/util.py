from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg


def get_perception_segmentation():
    #get config for Detector for inference (acts as perception module)
    cfgd = get_cfg()
   

    cfgd_str= "/home/iccs/Desktop/isense/hidrive/working/detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_C4_1x.yaml"
        
    cfgd_model_path = "https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_50_C4_1x/137259246/model_final_9243eb.pkl"
    cfgd.merge_from_file(cfgd_str)
    
    cfgd.MODEL.WEIGHTS =cfgd_model_path
    
    # cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = True
    
    cfgd.MODEL.DEVICE="cuda"

    detector = DefaultPredictor(cfgd)
    return detector