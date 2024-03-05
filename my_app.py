import hydra
from hydra.utils import instantiate
import logging
from omegaconf import DictConfig
import datetime
import os
from os.path import join
from torch.utils.data import DataLoader
from torch.utils.data import ChainDataset,ConcatDataset

from intention_prediction.data.prevention_data_iter import *
from intention_prediction.data.preprocess_labels import _preprocess_label_file
from intention_prediction.src.train_epochs import *
from intention_prediction.models.load_resnet import *
from intention_prediction.data.data_loader_utils import collate_fn_padding
from intention_prediction.data.video_segment_dataset import prevention_dataset_val , prevention_dataset_train, construct_ds

from itertools import cycle

log = logging.getLogger(__name__)


@hydra.main(config_path="/home/iccs/Desktop/isense/events/intention_prediction", config_name="/conf/conf.yaml")
def my_app(cfg: DictConfig):  
    logging.info("Initializing")


    # dataset_train = (read_frame_from_iter_train(path_to_video = "/home/iccs/Desktop/isense/events/intention_prediction/processed_data/video_camera1.mp4",
    #                                            path_to_label = "/home/iccs/Desktop/isense/events/intention_prediction/processed_data/detection_camera1/lane_changes_preprocessed.txt",
    #                                            prediction_horizon=5,
    #                                            splits=(0.8,0.1,0.1)))

    # dataset_val = read_frame_from_iter_val(path_to_video = "/home/iccs/Desktop/isense/events/intention_prediction/processed_data/video_camera1.mp4",
    #                                            path_to_label = "/home/iccs/Desktop/isense/events/intention_prediction/processed_data/detection_camera1/lane_changes_preprocessed.txt",
    #                                            prediction_horizon=5,
    #                                            splits=(0.8,0.1,0.1))

    dataset_train = prevention_dataset_train(root= "/home/iccs/Desktop/isense/events/intention_prediction/processed_data/segmented_frames",
                                             label_root="/home/iccs/Desktop/isense/events/intention_prediction/processed_data/detection_camera1/lane_changes.txt")

    weights=dataset_train.get_weights(),


    dataset_val = prevention_dataset_val(root= "/home/iccs/Desktop/isense/events/intention_prediction/processed_data/segmented_frames",
                                             label_root="/home/iccs/Desktop/isense/events/intention_prediction/processed_data/detection_camera1/lane_changes.txt")

    # dataset_test = read_frame_from_iter_test(path_to_video = "/home/iccs/Desktop/isense/events/intention_prediction/processed_data/video_camera1.mp4",
    #                                            path_to_label = "/home/iccs/Desktop/isense/events/intention_prediction/processed_data/detection_camera1/lane_changes_preprocessed.txt",
    #                                            prediction_horizon=5,
    #                                            splits=(0.8,0.1,0.1))
    ##print dataset statistics
    print(repr(dataset_train))

    dataset_train = ConcatDataset([prevention_dataset_train(root= "/home/iccs/Desktop/isense/events/intention_prediction/processed_data/segmented_frames",
                                             label_root="/home/iccs/Desktop/isense/events/intention_prediction/processed_data/detection_camera1/lane_changes.txt"),
                                prevention_dataset_train(root= "/home/iccs/Desktop/isense/events/intention_prediction/processed_data/new_data/processed_data_02/segmented_frames",
                                             label_root="/home/iccs/Desktop/isense/events/intention_prediction/processed_data/new_data/processed_data_02/processed_data/detection_camera1/lane_changes.txt"),
                                prevention_dataset_train(root= "/home/iccs/Desktop/isense/events/intention_prediction/processed_data/new_data/processed_data_03/segmented_frames",
                                             label_root="/home/iccs/Desktop/isense/events/intention_prediction/processed_data/new_data/processed_data_03/processed_data/detection_camera1/lane_changes.txt")
                                ])

    print("Final dataset size is {}".format(len(dataset_train)))

    # dataloader = instantiate(config = cfg.datasets.prevention_loader)  #recheck
    dataloader_train = DataLoader(dataset_train , batch_size=1 , collate_fn= collate_fn_padding , shuffle=True)
    dataloader_val = DataLoader(dataset_val , batch_size=1 , collate_fn= collate_fn_padding , shuffle=True)
    # dataloader_test = DataLoader(dataset_test , batch_size=1 , collate_fn= collate_fn_padding)

    # model = instantiate(cfg.conf.models)   #recheck 
    model = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=False , model_num_class=3)
    state_dict = torch.hub.load_state_dict_from_url(    "https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/kinetics/SLOWFAST_4x16_R50.pyth"
                                                    )
    # input(state_dict)
    # # state_dict_new = {k,v  for k,v in state_dict.items()}
    # model.load_state_dict(state_dict)

    # print(torch.hub.help('facebookresearch/pytorchvideo', 'x3d_s'))

    optimizer = torch.optim.Adam(params  = model.parameters() , lr=0.003)

    epochs = cfg.conf.trainer.epochs

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer= optimizer , gamma=0.9)

    dev = torch.device("cuda:0")

    writer = SummaryWriter(log_dir= "/home/iccs/Desktop/isense/events/intention_prediction/logs/run_{}".format(now:=datetime.datetime.now()) )
    
    
    train(cfg , writer=writer,
          dataloader_train = dataloader_train , 
          dataloader_val = dataloader_val,
          dataset_train = dataset_train,
          dataset_val = dataset_val,
          weights = weights ,
          model = model , 
          optimizer = optimizer,
          scheduler = scheduler,
          epochs = epochs,
          dev= dev,
          model_save_path="/home/iccs/Desktop/isense/events/intention_prediction/models/weights/train_01_03_04.pt",
          num_iterations_gr_accum = 16,
          log_dict = {"lr":0.003})
    
    # test(cfg , dataloader_test , model , dev)

if __name__=="__main__":
    """
    confg app / check preprocessing dependencies
    """
    if not os.path.isfile("/home/iccs/Desktop/isense/events/intention_prediction/processed_data/detection_camera1/lane_changes_preprocessed.txt"):
        _preprocess_label_file()
    
    my_app()