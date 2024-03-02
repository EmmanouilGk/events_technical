import contextlib
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm

from sklearn.metrics import precision_score,recall_score

class logging_utils():
    """
    logging helper methods for ML models info and debug
    """
    

    def __init__(self,
                 writer_path,
                 len_dataloader,
                 current_epoch) -> None:
      
        self._current_epoch = current_epoch


    def tb_writer(self, func):
        def wrapper(*args,**kwargs):
            losses = func(*args ,**kwargs)
            self.writer.addScalar(losses["desc"] , losses["val"] , losses["time"])
        return wrapper
    
    @property
    def epoch(self):
        print("Getting value...")
        return self._current_epoch

    @epoch.setter
    def epoch(self, value):
        print("Setting value...")
        self._current_epoch = value



def train(*args,**kwargs):
    """
    config batch training
    """
    max_epochs:int = kwargs["epochs"]  #max train epoch
    
    writer = SummaryWriter(log_dir= "/home/iccs/Desktop/isense/events/intention_prediction/logs" )
    scheduler = kwargs["scheduler"]
    criterion = torch.nn.CrossEntropyLoss()
    model=kwargs["model"]
    optimizer=kwargs["optimizer"]
    kwargs.update({"criterion":criterion})

    for epoch in  range(max_epochs):

        losses_dict = train_one_epoch(*args , **kwargs)

        for i,loss in enumerate(losses_dict["val"]):
            writer.add_scalar(losses_dict["desc"] , loss ,  (epoch-1)*losses_dict["batch_count"] + i)

        val_losses_dict = val_one_epoch(*args, **kwargs)

        writer.add_scalar(val_losses_dict["desc"] , val_losses_dict["val"] , (epoch-1)*max_batches + list(range(1,max_batches)))
        writer.add_scalar("Val Accuracy" , val_losses_dict["acc"] ,  (epoch-1)*max_batches + list(range(1,max_batches)))
        writer.add_scalar("Val Precision" , val_losses_dict["acc"] ,  (epoch-1)*max_batches + list(range(1,max_batches)))
        writer.add_scalar("Val Recall" , val_losses_dict["acc"] ,  (epoch-1)*max_batches + list(range(1,max_batches)))

        scheduler.step()
        torch.save({'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            
            }, kwargs["model_save_path"])


#equiv to logging_utils().tb_writer(train_one_epoch(*args,**kwargs))
# @logging_utils().tb_writer():
def train_one_epoch(*args , **kwargs):

        data_loader = kwargs["dataloader_train"]
        dev=kwargs["dev"]
        model = kwargs["model"]
        criterion = kwargs["criterion"]
        optimizer = kwargs["optimizer"]
        model = model.to(dev)
        loss_epoch = []
    #     for i in range(num_iterations):
    # accumulated_gradients = 0
        max_batches = 0
        for batch_idx , (frames , maneuver_type) in (pbar:=tqdm(enumerate(data_loader))): 

            pbar.set_description_str("Batch: {}".format(batch_idx))
            
            optimizer.zero_grad()

            frames = frames.to(dev)
            maneuver_type=maneuver_type.to(dev)

            prediction = model(frames)
            
            loss = criterion(prediction , maneuver_type)
            loss.backward()
            loss_epoch.append(loss.item())
            optimizer.step()
            pbar.set_postfix_str("Batch loss {:0.2f}".format(loss.item()))
            max_batches+=1


        return {"desc":"loss_train_epoch","val":np.array(loss_epoch),"batch_count":max_batches}

@torch.no_grad
def val_one_epoch(*args , **kwargs):
        """
        validate one epoch
        """
        data_loader = kwargs["dataloader_val"]
        dev=kwargs["dev"]
        model = kwargs["model"]
        criterion = kwargs["criterion"]

        loss_epoch = []
        predictions_epoch = []
        labels_epoch = []

        for batch_idx , (frames , maneuver_type) in (pbar:=tqdm(enumerate(data_loader))): 

            pbar.set_description_str("Batch: {}/{}".format(batch_idx))
            
            frames = frames.to(dev)
            maneuver_type=maneuver_type.to(dev)
            prediction = model(frames)
            prediction =torch.nn.Softmax(prediction)
           
            loss = criterion(prediction , maneuver_type)

            #to comute eval metrics
            predictions_epoch.append(prediction)
            labels_epoch.append(maneuver_type)

            loss_epoch.append(loss.item())

            pbar.set_postfix_str("Val Batch loss {:0.2f}".format(loss.item()))

        acc = [x == y for x,y in zip(predictions_epoch , labels_epoch)]
        
        pres=precision_score(labels_epoch , predictions_epoch)
        rec =recall_score(labels_epoch , predictions_epoch)

        return {"desc":"loss_train_epoch","val":np.array(loss_epoch),"acc":acc,"pres":pres , "rec":rec}