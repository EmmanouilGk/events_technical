import hydra
import logging
from omegaconf import DictConfig

from os.path import join
from torch.utils.data import DataLoader

from data.prevention_data_iter import *
from train.train_epochs import *
from models.load_resnet import *



log = logging.getLogger(__name__)


@hydra.main(config_path="/home/iccs/Desktop/isense/events/intention_prediction", config_name="conf")
def main(cfg: DictConfig):
    logging.info("Initializing")

    dataset = instantiate(config = cfg.datasets.prevention)
    
    model = instantiate(cfg.models.resnet)

    optimizer = instantiate(cfg.trainer.train, params  = model.params())

    train(cfg , dataset = dataset , model = model , optimizer = optimizer)

if __name__=="__main__":
    main()