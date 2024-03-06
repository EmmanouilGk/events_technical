import contextlib
from typing import Dict
import numpy as np
import torch
import datetime
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm

from sklearn.metrics import accuracy_score, precision_score,recall_score
from ..data.data_loader_utils import collate_fn_padding

from ..data.prevention_data_iter import read_frame_from_iter_train, read_frame_from_iter_val

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
    config batch training and validtion
    expexted kwargs:
     writer,scheduler,model,optimizer,dataloader (train and val), device, save path model weights
    """
    max_epochs:int = kwargs["epochs"]  #max train epoch
    dev = kwargs["dev"]
    now = datetime.datetime.now()
    scheduler: torch.optim.lr_scheduler.ExponentialLR = kwargs["scheduler"]

    #load weights for training
    weights=kwargs["weights"][0]
    weights = [weights["LK"],weights["LLC"],weights["RLC"],]


    criterion = torch.nn.CrossEntropyLoss( weight= torch.tensor(data = ( weights[0] , weights[1], weights[2]), dtype=torch.float , device=dev))
    model=kwargs["model"]
    optimizer=kwargs["optimizer"]

    ##update config params
    kwargs.update({"criterion":criterion})
    class_map=dict({0:"LK",1:"LLC",2:"RLC"})

   
    writer= kwargs["writer"]
    
    #train and val
    for epoch in  range(max_epochs):

        losses_dict = train_one_epoch(*args , **kwargs)   #train losses dict

        for i,loss in enumerate(losses_dict["val"]):
            writer.add_scalar(losses_dict["desc"] , loss ,  (epoch-1)*losses_dict["batch_count"] + i)  #plot losses 

        val_losses_dict = val_one_epoch(*args, **kwargs)  #vla losses dict

        for i, (desc , val) in enumerate(val_losses_dict.items()):
            if desc=="loss_val_epoch": 
                 for i in range(val.shape[0]):
                    writer.add_scalar("Val Batch_Loss" , val[i] , (epoch-1)*losses_dict["batch_count"] + i)  #plot val loss
                 continue
            if desc=="val_pres":
                for i in range(2):
                    writer.add_scalars("Val micro" , {"Class {}".format(class_map[i]):val[1][i]},  i)   # add epoch wise acc,pres etc metrics 

                writer.add_scalar("Val macro" , val[0],  epoch)   # add epoch wise acc,pres etc metrics 


        scheduler.step()

        # writer.add_hparams({"lr" : scheduler.get_last_lr() } ,
        #                     {"loss_mean_val":np.mean(val_losses_dict["loss_val_epoch"])})

        torch.save({'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            
            }, kwargs["model_save_path"])
        
        

        #reset datasets for multi-epoch iterations ->change again
        dataset_train = (preven(path_to_video = "/home/iccs/Desktop/isense/events/intention_prediction/processed_data/video_camera1.mp4",
                                            path_to_label = "/home/iccs/Desktop/isense/events/intention_prediction/processed_data/detection_camera1/lane_changes_preprocessed.txt",
                                            prediction_horizon=5,
                                            splits=(0.8,0.1,0.1)))
        
        kwargs["dataloader_train"]=DataLoader(dataset_train , batch_size=1 , collate_fn= collate_fn_padding , )
        
        dataset_val = read_frame_from_iter_val(path_to_video = "/home/iccs/Desktop/isense/events/intention_prediction/processed_data/video_camera1.mp4",
                                                path_to_label = "/home/iccs/Desktop/isense/events/intention_prediction/processed_data/detection_camera1/lane_changes_preprocessed.txt",
                                                prediction_horizon=5,
                                                splits=(0.8,0.1,0.1))
        
        kwargs["dataloader_val"]=DataLoader(dataset_val , batch_size=1 , collate_fn= collate_fn_padding , )


#equiv to logging_utils().tb_writer(train_one_epoch(*args,**kwargs))
# @logging_utils().tb_writer():
def train_one_epoch(*args , **kwargs):

        data_loader = kwargs["dataloader_train"]
        dev=kwargs["dev"]
        model = kwargs["model"].to(dev)
        criterion = kwargs["criterion"]
        optimizer = kwargs["optimizer"]
        writer= kwargs["writer"]
        loss_epoch = []
        
        accumulated_gradients = kwargs["num_iterations_gr_accum"]

        max_batches = 0
        predictions_epoch=[]
        labels_epoch=[]

        _debug_counter=0
        _debug_max=1

        for batch_idx , (frames , maneuver_type) in (pbar:=tqdm(enumerate(data_loader))): 

            pbar.set_description_str("Train Batch: {}".format(batch_idx))
            
            frames = frames.to(dev)
            maneuver_type=maneuver_type.to(dev)

            prediction = model(frames)

            loss = criterion(prediction , maneuver_type) / accumulated_gradients

            loss.backward()

            loss_epoch.append(loss.item())
            writer.add_scalar("Online batch loss",loss_epoch[-1],batch_idx)

            if ((batch_idx + 1) % accumulated_gradients == 0):
                optimizer.step()
                optimizer.zero_grad()
                max_batches+=1
                writer.add_scalar("Online batch loss-During update ", loss_epoch[-1] , batch_idx)

            pbar.set_postfix_str("Batch loss {:0.2f}".format(loss.item()))

            predictions_epoch.append(prediction.detach().cpu().numpy())
            labels_epoch.append(maneuver_type.detach().cpu().numpy())

            _debug_counter+=1
            if _debug_counter==_debug_max:break

        acc = np.mean([x == y for x,y in zip(map(lambda x: np.argmax(x) , predictions_epoch) , labels_epoch)])
        
        return {"desc":"loss_train_epoch","val":np.array(loss_epoch),"batch_count":max_batches}

@torch.no_grad
def val_one_epoch(*args , **kwargs)->Dict:
        """
        validate one epoch
        """
        data_loader = kwargs["dataloader_val"]
        dev=kwargs["dev"]
        model = kwargs["model"].to(dev)
        criterion = kwargs["criterion"]

        loss_epoch = []
        predictions_epoch = []
        labels_epoch = []
        max_epochs_val = 0

        
        for batch_idx , (frames , maneuver_type) in (pbar:=tqdm(enumerate(data_loader))): 
            pbar.set_description_str("Val Batch: {}".format(batch_idx))
            
            frames = frames.to(dev) 
            maneuver_type=maneuver_type.type(torch.LongTensor).to(dev)
            

            prediction = model(frames)
           
            loss = criterion(prediction, maneuver_type)

            #to comute eval metrics
            predictions_epoch.append(prediction.detach().cpu().numpy())
            labels_epoch.append(maneuver_type.detach().cpu().numpy())

            loss_epoch.append(loss.item())

            pbar.set_postfix_str("Val Batch loss {:0.2f}".format(loss.item()))

            max_epochs_val+=1

        #convert to int categorical labels
        predictions_epoch=list(map(lambda x: np.argmax(x) , predictions_epoch))
        labels_epoch=list(map(lambda x: int(x) , labels_epoch))
        
        input(predictions_epoch)
        input(labels_epoch)
        
        acc = accuracy_score(labels_epoch , predictions_epoch)
        pres_avg=precision_score(labels_epoch , predictions_epoch , average = "macro")
        pres_class=precision_score(labels_epoch , predictions_epoch , average = None)
        
        rec =recall_score(labels_epoch , predictions_epoch , average="macro")
        rec_class = recall_score(labels_epoch , predictions_epoch , average= None)

        print(pres_class)
        return {"loss_val_epoch":np.array(loss_epoch),
                "val_acc":acc,
                "val_pres":[pres_avg , pres_class] , 
                "val_rec":[rec,rec_class] ,
                "batch_count":max_epochs_val}