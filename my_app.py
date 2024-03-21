import fnmatch
import hydra
from hydra.utils import instantiate
import logging
from omegaconf import DictConfig
import datetime
from pytorchvideo.models.slowfast import create_slowfast
from pytorchvideo.models.head import create_res_basic_head
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg

import os
from os.path import join
from torch.utils.data import DataLoader
from torch.utils.data import ChainDataset,ConcatDataset
from omegaconf import OmegaConf 
from intention_prediction.data.prevention_data_iter import *
from intention_prediction.data.preprocess_labels import _preprocess_label_file
from intention_prediction.src.train_epochs import *
from intention_prediction.src.test import *
from intention_prediction.models.load_resnet import *
from intention_prediction.data.data_loader_utils import collate_fn_padding
from intention_prediction.data.video_segment_dataset import (_get_semented_data_paths, custom_concat_dataset, prevention_dataset_val , prevention_dataset_train, 
                                                               construct_ds, compute_weights , compute_weights_binary_cls,prevention_dataset_test, union_prevention)

from itertools import cycle

log = logging.getLogger(__name__)


@hydra.main(config_path="/home/iccs/Desktop/isense/events/intention_prediction", config_name="/conf/conf.yaml")
def my_app(cfg: DictConfig):  
    logging.info("Initializing")

    cfgv = OmegaConf.to_container(cfg)
    
    _root_dir = "/home/iccs/Desktop/isense/events/intention_prediction/processed_data/"
    
    # dataset_kwargs = _get_data_conf_(recording = , drive = )
    dataset_train =  ConcatDataset([
                                    # prevention_dataset_train(root= join(_root_dir , "new_data/recording_05/drive_03/segmented_frames"),
                                    #          label_root=join(_root_dir , "new_data/recording_05/drive_03/processed_data/detection_camera1/lane_changes.txt"),
                                    #          gt_root=join(_root_dir , "new_data/recording_05/drive_03/processed_data/detection_camera1/labels.txt"),
                                    #          detection_root = join(_root_dir , "new_data/recording_05/drive_03/processed_data/detection_camera1/detections_tracked.txt")
                                    #          ,desc = "Rec_05_03"),
                                    #     ,  
    
                                    # prevention_dataset_train(root= join(_root_dir , "new_data/recording_04/drive_01/segmented_frames"),
                                    #          label_root=join(_root_dir , "new_data/recording_04/drive_01/processed_data/detection_camera1/lane_changes.txt"),
                                    #          gt_root=join(_root_dir , "new_data/recording_04/drive_01/processed_data/detection_camera1/labels.txt"),
                                    #          detection_root = join(_root_dir , "new_data/recording_04/drive_01/processed_data/detection_camera1/detections_tracked.txt")
                                    #          ,desc = "Rec_04_01"),
    

                                    prevention_dataset_train(root= join(_root_dir , "new_data/recording_02/drive_01/segmented_frames"),
                                             label_root=join(_root_dir , "new_data/recording_02/drive_01/processed_data/detection_camera1/lane_changes.txt"),
                                             gt_root=join(_root_dir , "new_data/recording_02/drive_01/processed_data/detection_camera1/labels.txt"),
                                             detection_root = join(_root_dir , "new_data/recording_02/drive_01/processed_data/detection_camera1/detections_tracked.txt")
                                             ,desc = "Rec_02_01",split=0.8),
    
                                    # prevention_dataset_train(root= join(_root_dir , "new_data/recording_05/drive_03/segmented_frames"),
                                    #          label_root=join(_root_dir , "new_data/recording_05/drive_03/processed_data/detection_camera1/lane_changes.txt"),
                                    #          gt_root=join(_root_dir , "new_data/recording_05/drive_03/processed_data/detection_camera1/labels.txt"),
                                    #          detection_root = join(_root_dir , "new_data/recording_05/drive_03/processed_data/detection_camera1/detections_tracked.txt")
                                    #          ,desc = "Rec_03_02"),
                                                               
                                    ])
    
    dataset_val = ConcatDataset([
                                    # prevention_dataset_val(root= join(_root_dir , "new_data/recording_05/drive_03/segmented_frames"),
                                    #          label_root=join(_root_dir , "new_data/recording_05/drive_03/processed_data/detection_camera1/lane_changes.txt"),
                                    #          gt_root=join(_root_dir , "new_data/recording_05/drive_03/processed_data/detection_camera1/labels.txt"),
                                    #          detection_root = join(_root_dir , "new_data/recording_05/drive_03/processed_data/detection_camera1/detections_tracked.txt")
                                    #          ),  
    
                                    # prevention_dataset_val(root= join(_root_dir , "new_data/recording_04/drive_01/segmented_frames"),
                                    #          label_root=join(_root_dir , "new_data/recording_04/drive_01/processed_data/detection_camera1/lane_changes.txt"),
                                    #          gt_root=join(_root_dir , "new_data/recording_04/drive_01/processed_data/detection_camera1/labels.txt"),
                                    #          detection_root = join(_root_dir , "new_data/recording_04/drive_01/processed_data/detection_camera1/detections_tracked.txt")
                                    #          ),
    
                                    prevention_dataset_val(root= join(_root_dir , "new_data/recording_02/drive_01/segmented_frames"),
                                             label_root=join(_root_dir , "new_data/recording_02/drive_01/processed_data/detection_camera1/lane_changes.txt"),
                                             gt_root=join(_root_dir , "new_data/recording_02/drive_01/processed_data/detection_camera1/labels.txt"),
                                             detection_root = join(_root_dir , "new_data/recording_02/drive_01/processed_data/detection_camera1/detections_tracked.txt")
                                             ,split=0.8),
                                    # prevention_dataset_val(root= join(_root_dir , "new_data/recording_05/drive_03/segmented_frames"),
                                    #          label_root=join(_root_dir , "new_data/recording_05/drive_03/processed_data/detection_camera1/lane_changes.txt"),
                                    #          gt_root=join(_root_dir , "new_data/recording_05/drive_03/processed_data/detection_camera1/labels.txt"),
                                    #          detection_root = join(_root_dir , "new_data/recording_05/drive_03/processed_data/detection_camera1/detections_tracked.txt")
                                    #          ) 
                                    ])
    
    dataset_test = ConcatDataset([
                                    # prevention_dataset_test(root= "/home/iccs/Desktop/isense/events/intention_prediction/processed_data/segmented_test_frames",
                                    #        label_root="/home/iccs/Desktop/isense/events/intention_prediction/processed_data/new_data/recording_04/drive_03/processed_data/detection_camera1/lane_changes.txt"),
                                   prevention_dataset_test(root= "/home/iccs/Desktop/isense/events/intention_prediction/processed_data/new_data/recording_02/drive_01/segmented_frames/",
                                            label_root="/home/iccs/Desktop/isense/events/intention_prediction/processed_data/new_data/recording_02/drive_01/processed_data/detection_camera1/lane_changes.txt",
                                            gt_root=join(_root_dir , "new_data/recording_02/drive_01/processed_data/detection_camera1/labels.txt"),split=0.8)
                                ])

    # ds_union = union_prevention()
    

    # dataloader = instantiate(config = cfg.datasets.prevention_loader)  #recheck
    if cfg.conf.use_weights:
        save_path="/home/iccs/Desktop/isense/events/intention_prediction/data/weights_torch/weights_union_prevention_binary.pt"
        if os.path.isfile(save_path):
            weights=torch.load(save_path)
            # assert len(weights)==len(dataset_train),"print {} {}".format(len(weights) , len(dataset_train))
        else: weights , class_w, weights_dict = compute_weights_binary_cls(ds = dataset_train,custom_scaling=10 , save_path =save_path )
        print("Final dataset size is {}".format(len(dataset_train)))
        dataloader_train = DataLoader(dataset_train , batch_size=1 , 
                                  collate_fn= collate_fn_padding , shuffle=False , 
                                  sampler =torch.utils.data.WeightedRandomSampler(weights=weights, num_samples = len(dataset_train),replacement=True),
                                  pin_memory=True)
        
        

    elif not cfg.conf.use_weights:
        dataloader_train = DataLoader(dataset_train , batch_size=1, 
                                  collate_fn= collate_fn_padding , shuffle=True)

   
    
    dataloader_val = DataLoader(dataset_val , batch_size=1 , collate_fn= collate_fn_padding , shuffle=True)


    dataloader_test = DataLoader(dataset_test , batch_size=1 , collate_fn= collate_fn_padding)
    
    model = create_slowfast(model_num_class =2 )


    # model = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=True )
    # print(model)
    # print(model.blocks)

    # ##define new model
    # for i,module in enumerate(model.blocks):
    #     print("a")
    #     if i==5:
    #         new_module_head = torch.nn.Sequential(*list(module.children())[:-2],
    #                                      torch.nn.Linear(in_features=2048,out_features=2,bias=True),
    #                                      torch.nn.AdaptiveAvgPool3d(output_size=1))
            
    # # model = torch.nn.Sequential(*list(model.blocks)[:-1] , new_module_head)
    
    # model = torch.nn.Sequential(*list(model.blocks)[:-1]  , create_res_basic_head(in_features = 2048, out_features = 2))

    print(model)
    input("waiting")
    
    optimizer = torch.optim.SGD(params  = model.parameters() , lr=0.01 , weight_decay = 0.0000001)

    epochs = cfg.conf.trainer.epochs

    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer= optimizer , gamma=0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,T_max=1.6)

    dev = torch.device("cuda:0")

    cfgd = get_cfg()
    # add_deeplab_config(cfg)
    # add_maskformer2_config(cfg)

    cfgd_str= "/home/iccs/Desktop/isense/hidrive/working/detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_C4_1x.yaml"
        
    cfgd_model_path = "https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_50_C4_1x/137259246/model_final_9243eb.pkl"
    cfgd.merge_from_file(cfgd_str)
    
    cfgd.MODEL.WEIGHTS =cfgd_model_path
    
    # cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = True
    
    cfgd.MODEL.DEVICE="cuda"

    detector = DefaultPredictor(cfgd)

    #==============================================================TRAIN-VAL-TEST================================================================
    name="slowfast_r50_pytorchvideo"
    writer = SummaryWriter(log_dir= "/home/iccs/Desktop/isense/events/intention_prediction/logs2/run_{}-{}".format(now:=datetime.datetime.now(),name) )
    print(cfgv)
    if cfgv["conf"]["mode"]=="train":
            train(cfg , writer=writer,
                dataloader_train = dataloader_train , 
                dataloader_val = dataloader_val,
                dataset_train = dataset_train,
                dataset_val = dataset_val,
                # weights = weights ,
                model = model , 
                optimizer = optimizer,
                scheduler = scheduler,
                epochs = epochs,
                dev= dev,
                model_save_path="/home/iccs/Desktop/isense/events/intention_prediction/models/weights/slowfast_r50_pytorchvideo.pt",
                # load_saved_model ="/home/iccs/Desktop/isense/events/intention_prediction/models/weights/train_01_03_05_r53r41_r2_1_b.pt",
                num_iterations_gr_accum = 5,
                log_dict = {"lr":0.003},
                detector = detector
                )
        
    
    test(cfg , 
        writer=writer,
        dataloader_test= dataloader_test,
        # weights = weights ,
        model = model , 
        epochs = epochs,
        dev= dev,
        load_saved_model ="/home/iccs/Desktop/isense/events/intention_prediction/models/weights/slowfast_r50_pytorchvideo.pt")

@torch.no_grad()
def construct_inference_video(
                              model,
                              srcp="/home/iccs/Desktop/isense/events/intention_prediction/processed_data/video_test.avi",
                              dstp="/home/iccs/Desktop/isense/events/intention_prediction/outputs/output.avi",
                              ):
    cap_in =cv2.VideoCapture(srcp)
    fps = cap_in.get(cv2.CAP_PROP_FPS)
    H = cap_in.get(cv2.CAP_PROP_FRAME_HEIGHT)
    W=cap_in.get(cv2.CAP_PROP_FRAME_WIDTH)
    total_frames = cap_in.get(cv2.CAP_PROP_MAX_FRAMES)

    cap_in.set(cv2.CAP_PROP_POS_FRAMES , total_frames*0.9)

    fourcc = cv2.VideoWriter_fourcc(*"MPEG")
    cap = cv2.VideoWriter(dstp , fourcc,fps,(W,H))

    with tqdm(total=0.1*total_frames):
        frame_list = []
        for idx in range(0,prediction_window:=30):
            ret,frame=cap_in.read()
            frame_list.append(frame)
        
        frame_tensor = inference_transform(frame)
        predictions = model(frame_list)
        img = frame_tensor.detach().cpu().numpy()
        
        img = cv2.writeText(img , "online prediction is {}".format(np.argmax(predictions)))

        cap.write(img)


if __name__=="__main__":
    """
    confg app / check preprocessing dependencies
    """
    if not os.path.isfile("/home/iccs/Desktop/isense/events/intention_prediction/processed_data/detection_camera1/lane_changes_preprocessed.txt"):
        _preprocess_label_file()
    
    my_app()