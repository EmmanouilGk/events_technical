import contextlib
from typing import Dict
import numpy as np
import torch
import datetime
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm

from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score,recall_score
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


def _write_val_values(val_losses_dict , writer , epoch , losses_dict ):
      for i, (desc , val) in enumerate(val_losses_dict.items()):
            if desc=="loss_val_epoch": 
                 for i in range(val.shape[0]):
                    writer.add_scalar("Val Batch_Loss" , val[i] , (epoch-1)*losses_dict["batch_count"] + i)  #plot val loss
                 continue
            if desc=="val_pres":
                for i in range(2):
                    writer.add_scalars("Val micro (0=Lk,1=Llc,2=Rlc)" , {"Class {}".format(i):val[1]},  i)   # add epoch wise acc,pres etc metrics 
                writer.add_scalar("Val Macro" , val[0],  epoch)   # add epoch wise acc,pres etc metrics 

                continue
            if desc =="val_pres_weighted":
                for i in range(2):
                    print(val)
                    writer.add_scalars("val_pres_weighted" , {"Class {}".format(i):val[1]},  i)   # add epoch wise acc,pres etc metrics 
                continue
            if desc=="val_rec":
                for i in range(2):
                    writer.add_scalars("Rec micro" , {"Class {}".format(i):val[1]},  i)   # add epoch wise acc,pres etc metrics 

                writer.add_scalar("Rec macro" , val[0],  epoch)   # add epoch wise acc,pres etc metrics 
                continue
        
            if desc=="acc":
                writer.add_scalar(desc , val,  epoch)   # add epoch wise acc,pres etc metrics 


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

    # #load weights for training
    # weights=kwargs["weights"]
    # print(weights)
    # # weights = [weights["LK"],weights["LLC"],weights["RLC"],]


    # criterion = torch.nn.CrossEntropyLoss( weight= torch.tensor(data = ( weights[0] , weights[1], weights[2]), dtype=torch.float , device=dev))
    criterion=torch.nn.CrossEntropyLoss()
    model=kwargs["model"]
    

    ##update config params
    kwargs.update({"criterion":criterion})
    class_map=dict({0:"LK",
                    1:"LLC",
                    2:"RLC"})

   
    writer= kwargs["writer"]
    
    torch.backends.cudnn.benchmark = True

    optimizer=kwargs["optimizer"]
    current_epoch=0
    if kwargs["load_saved_model"]:
        print("Loading saved model at path {}".format(kwargs["load_saved_model"]))
        checkpoint = torch.load(kwargs["load_saved_model"])
        
        model.load_state_dict(checkpoint["model_state_dict"])
        model=model.to(dev)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        current_epoch = checkpoint["epoch"]
        
    

    #train and val
    for epoch in  range(0 , max_epochs-current_epoch):

        losses_dict = train_one_epoch(*args , **kwargs)   #train losses dict

        for i,loss in enumerate(losses_dict["val"]):
            writer.add_scalar(losses_dict["desc"] , loss ,  (epoch-1)*losses_dict["batch_count"] + i)  #plot losses 

        writer.add_scalar("Accuracy" , losses_dict["Epoch_mean_Accuracy"]  , epoch)
        
        torch.save({'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            "loss" : losses_dict["desc"][-1],
            }, kwargs["model_save_path"])

        val_losses_dict = val_one_epoch(*args, **kwargs)  #vla losses dict
        
        _write_val_values(val_losses_dict=val_losses_dict , writer=writer,epoch=epoch,losses_dict=losses_dict)
      

        scheduler.step()

        # writer.add_hparams({"lr" : scheduler.get_last_lr() } ,
        #                     {"loss_mean_val":np.mean(val_losses_dict["loss_val_epoch"])})

        torch.save({'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            
            }, kwargs["model_save_path"])
        

def train_one_epoch(*args , **kwargs):
        """
        train batches logic
        """
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


        acc = accuracy_score(labels_epoch , s:=list(map(lambda x:np.argmax(x) , predictions_epoch))  )  

        return {"desc":"loss_train_epoch",
                "val":np.array(loss_epoch),
                "batch_count":max_batches,
                "Epoch_mean_Accuracy" : acc}

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

        
        acc = accuracy_score(labels_epoch , predictions_epoch)
        bacc = balanced_accuracy_score(labels_epoch , predictions_epoch)

        pres_avg=precision_score(labels_epoch , predictions_epoch , average = "macro")
        pres_class=precision_score(labels_epoch , predictions_epoch , average = "micro")
        pres=precision_score(labels_epoch , predictions_epoch , average = "weighted")

        rec =recall_score(labels_epoch , predictions_epoch , average="macro")
        rec_class = recall_score(labels_epoch , predictions_epoch , average= "micro")


        print(pres_class)
        return {"loss_val_epoch":np.array(loss_epoch),
                "val_acc":acc,
                "val_pres":[pres_avg , pres_class] , 
                "val_rec":[rec,rec_class] ,
                "batch_count":max_epochs_val , 
                "val_pres_global":pres,
                "bacc":bacc}