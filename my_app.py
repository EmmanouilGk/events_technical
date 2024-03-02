import hydra
from hydra.utils import instantiate
import logging
from omegaconf import DictConfig

import os
from os.path import join
from torch.utils.data import DataLoader

from data.prevention_data_iter import *
from data.preprocess_labels import _preprocess_label_file
from src.train_epochs import *
from models.load_resnet import *
from data.data_loader_utils import collate_fn_padding



log = logging.getLogger(__name__)


@hydra.main(config_path="/home/iccs/Desktop/isense/events/intention_prediction", config_name="/conf/conf.yaml")
def my_app(cfg: DictConfig):  
    logging.info("Initializing")

    # dataset = instantiate(config = cfg.conf.datasets)  #recheck

    # dataset_prevention = base_class_prevention(path_to_video = "/home/iccs/Desktop/isense/events/intention_prediction/processed_data/video_train.avi",
    #                                            path_to_label = "/home/iccs/Desktop/isense/events/intention_prediction/processed_data/detection_camera1/lane_changes_preprocessed.txt",
    #                                            prediction_horizon=5,
    #                                            splits=(0.8,0.1,0.1))


    dataset_train = read_frame_from_iter_train(path_to_video = "/home/iccs/Desktop/isense/events/intention_prediction/processed_data/video_train.avi",
                                               path_to_label = "/home/iccs/Desktop/isense/events/intention_prediction/processed_data/detection_camera1/lane_changes_preprocessed.txt",
                                               prediction_horizon=5,
                                               splits=(0.8,0.1,0.1))

    dataset_val = read_frame_from_iter_val(path_to_video = "/home/iccs/Desktop/isense/events/intention_prediction/processed_data/video_train.avi",
                                               path_to_label = "/home/iccs/Desktop/isense/events/intention_prediction/processed_data/detection_camera1/lane_changes_preprocessed.txt",
                                               prediction_horizon=5,
                                               splits=(0.8,0.1,0.1))

    dataset_test = read_frame_from_iter_test(path_to_video = "/home/iccs/Desktop/isense/events/intention_prediction/processed_data/video_train.avi",
                                               path_to_label = "/home/iccs/Desktop/isense/events/intention_prediction/processed_data/detection_camera1/lane_changes_preprocessed.txt",
                                               prediction_horizon=5,
                                               splits=(0.8,0.1,0.1))

    # dataloader = instantiate(config = cfg.datasets.prevention_loader)  #recheck
    dataloader_train = DataLoader(dataset_train , batch_size=1 , collate_fn= collate_fn_padding)
    dataloader_val = DataLoader(dataset_val , batch_size=1 , collate_fn= collate_fn_padding)
    dataloader_test = DataLoader(dataset_test , batch_size=1 , collate_fn= collate_fn_padding)

    # model = instantiate(cfg.conf.models)   #recheck 
    model = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=True)

    optimizer = torch.optim.Adam(params  = model.parameters() , lr=0.003)

    epochs = cfg.conf.trainer.epochs

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer= optimizer , gamma=0.9)

    dev = torch.device("cuda:0")

    train(cfg , 
          dataloader_train = dataloader_train , 
          dataloader_val = dataloader_val,
          model = model , 
          optimizer = optimizer,
          scheduler = scheduler,
          epochs = epochs,
          dev= dev,
          model_save_path="/home/iccs/Desktop/isense/events/intention_prediction/models/weights/")
    
    test(cfg , dataloader_test , model , dev)

if __name__=="__main__":
    """
    confg app / check preprocessing dependencies
    """
    if not os.path.isfile("/home/iccs/Desktop/isense/events/intention_prediction/processed_data/detection_camera1/lane_changes_preprocessed.txt"):
        _preprocess_label_file()
    
    my_app()