import hydra
from hydra.utils import instantiate
import logging
from omegaconf import DictConfig

from os.path import join
from torch.utils.data import DataLoader

from data.prevention_data_iter import *
from src.train_epochs import *
from models.load_resnet import *



log = logging.getLogger(__name__)


@hydra.main(config_path="/home/iccs/Desktop/isense/events/intention_prediction", config_name="/conf/conf.yaml")
def my_app(cfg: DictConfig):  
    logging.info("Initializing")

    input(cfg)

    # dataset = instantiate(config = cfg.conf.datasets)  #recheck

    dataset = read_frame_from_iter(path_to_video= "/home/iccs/Desktop/isense/events/intention_prediction/processed_data/video_train.avi",
                                  path_to_label="/home/iccs/Desktop/isense/events/intention_prediction/processed_data/detection_camera1/lane_changes.txt")

    # dataloader = instantiate(config = cfg.datasets.prevention_loader)  #recheck
    dataloader = DataLoader(dataset , batch_size=3)
    
    # model = instantiate(cfg.conf.models)   #recheck 
    model = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=True)

    optimizer = torch.optim.Adam(params  = model.parameters() , lr=0.003)

    epochs = cfg.conf.trainer.epochs

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer= optimizer , gamma=0.9)

    dev = torch.device("cuda:0")

    train(cfg , 
          dataloader = dataloader , 
          model = model , 
          optimizer = optimizer,
          scheduler = scheduler,
          epochs = epochs,
          dev= dev)

if __name__=="__main__":
    my_app()